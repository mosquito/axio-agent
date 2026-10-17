"""Protected files and locks for local credential backends."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from .types import CredentialError, complete_operation

if os.name == "posix":
    import fcntl


def check_file(info: os.stat_result, *, directory: bool = False) -> None:
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    protected = not (info.st_mode & 0o022) if directory else stat.S_IMODE(info.st_mode) == 0o600
    if not expected or info.st_uid != os.getuid() or not protected:
        raise CredentialError(
            "Credential paths must be owned by this user, files mode 0600, directories not writable by others."
        )
    if not directory and info.st_nlink != 1:
        raise CredentialError("Credential files must not have hard links.")


def open_file(path: Path, flags: int) -> int:
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
    try:
        check_file(os.fstat(fd))
    except BaseException:
        os.close(fd)
        raise
    return fd


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileNotFoundError:
        make_directory(path.parent)
        make_directory(path)
    except FileExistsError:
        pass
    else:
        sync_directory(path.parent)


class ProtectedDirectory:
    def __init__(self, path: Path) -> None:
        if os.name != "posix":
            raise CredentialError("Local credential stores require POSIX permissions and file locks.")
        self.path = path.absolute()

    def prepare(self) -> None:
        try:
            make_directory(self.path)
            check_file(self.path.lstat(), directory=True)
        except OSError:
            raise CredentialError("Could not open the protected credential directory.") from None

    def read(self, name: str) -> str | None:
        self.prepare()
        try:
            fd = open_file(self.path / name, os.O_RDONLY)
        except FileNotFoundError:
            return None
        except OSError:
            raise CredentialError("Could not open the protected credential file.") from None
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                return stream.read()
        except (OSError, UnicodeError):
            raise CredentialError("Could not read the credential file.") from None

    def write(self, name: str, raw: str) -> None:
        self.prepare()
        temporary: str | None = None
        try:
            destination = self.path / name
            try:
                check_file(destination.lstat())
            except FileNotFoundError:
                pass
            fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=self.path)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            sync_directory(self.path)
        except OSError:
            raise CredentialError("Could not commit the credential file.") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        await complete_operation(asyncio.to_thread(self.prepare))
        try:
            fd = open_file(self.path / (key + ".lock"), os.O_RDWR | os.O_CREAT)
        except OSError:
            raise CredentialError("Could not open the credential lock.") from None
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise CredentialError("Could not acquire the credential lock.") from None
                    await asyncio.sleep(0.025)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def blocking_lock(self, key: str) -> Iterator[None]:
        """Hold a short file-update lock from a worker thread."""
        self.prepare()
        try:
            fd = open_file(self.path / (key + ".lock"), os.O_RDWR | os.O_CREAT)
        except OSError:
            raise CredentialError("Could not open the credential update lock.") from None
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                raise CredentialError("Could not acquire the credential update lock.") from None
            yield
        finally:
            os.close(fd)
