import asyncio
import os
import signal
import time
from collections.abc import AsyncGenerator

from axio.field import StrictStr


def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and its entire process group."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def _format_records(records: list[tuple[float, str, str]]) -> str:
    """Merge consecutive same-stream records within 0.5s into log entries.

    Produces structured output so the model sees stdout vs stderr with
    timing: ``[00:01.234 stderr] something went wrong``.
    """
    if not records:
        return "(no output)"

    # (first_ts, last_ts, key, accumulated_text)
    merged: list[tuple[float, float, str, str]] = []
    for ts, key, text in records:
        if merged and merged[-1][2] == key and (ts - merged[-1][1]) <= 0.5:
            prev = merged[-1]
            merged[-1] = (prev[0], ts, key, prev[3] + text)
        else:
            merged.append((ts, ts, key, text))

    lines: list[str] = []
    for first_ts, _, key, text in merged:
        mins, secs = divmod(first_ts, 60)
        header = f"[{int(mins):02d}:{secs:06.3f} {key}]"
        lines.append(f"{header} {text.rstrip(chr(10))}")
    return "\n".join(lines)


async def _shell_stream(
    command: str,
    timeout: int = 5,
    cwd: str = ".",
    stdin: str | None = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Yield ``(key, text)`` tuples where *key* is ``"stdout"`` or ``"stderr"``."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        yield ("stderr", f"[error: {exc}]")
        return

    if stdin is not None:
        assert proc.stdin is not None
        proc.stdin.write(stdin.encode())
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    assert proc.stderr is not None

    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

    async def _read_pipe(pipe: asyncio.StreamReader, key: str) -> None:
        while True:
            line = await pipe.readline()
            if not line:
                break
            await queue.put((key, line.decode(errors="replace")))
        await queue.put(None)

    stdout_task = asyncio.create_task(_read_pipe(proc.stdout, "stdout"))
    stderr_task = asyncio.create_task(_read_pipe(proc.stderr, "stderr"))

    timed_out = False
    deadline = time.monotonic() + timeout

    try:
        done_count = 0
        while done_count < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except TimeoutError:
                timed_out = True
                break
            if item is None:
                done_count += 1
            else:
                yield item

        if not timed_out:
            await asyncio.gather(stdout_task, stderr_task)
    finally:
        if timed_out:
            _kill_process(proc)
            stdout_task.cancel()
            stderr_task.cancel()
            await proc.wait()

    if timed_out:
        yield ("stderr", f"[timeout: command exceeded {timeout}s]")
        return

    returncode = await proc.wait()
    if returncode != 0:
        yield ("stderr", f"[exit code: {returncode}]")


async def shell(
    command: StrictStr,
    timeout: int = 5,
    cwd: StrictStr = ".",
    stdin: str | None = None,
) -> str:
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. The default timeout is 5s — raise it for long-running
    commands (installs, builds, and test suites often need 60-300s).
    Avoid interactive commands."""
    records: list[tuple[float, str, str]] = []
    t0 = time.monotonic()
    async for key, text in _shell_stream(command, timeout, cwd, stdin):
        records.append((time.monotonic() - t0, key, text))
    return _format_records(records)


# Streaming hooks consumed by axio.tool.Tool.call_streaming / format_stream_result.
shell.stream = _shell_stream  # type: ignore[attr-defined]
shell.format_stream_result = staticmethod(_format_records)  # type: ignore[attr-defined]
