"""Build entry-point for the ``webrtc_apm`` C++ extension.

The pyproject.toml only declares static metadata; setuptools needs a
setup.py to wire up an ``Extension`` (PEP 621 has no declarative form
for C/C++ ext_modules with non-trivial flags).  Discover the
webrtc-audio-processing-1 include + library paths via pkg-config so
this works on any distro that ships the upstream package (e.g. Arch:
``webrtc-audio-processing``, Debian/Ubuntu: ``libwebrtc-audio-processing-1-dev``).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext  # type: ignore[import-not-found]
from setuptools import setup  # type: ignore[import-untyped]


def _pkg_config(*args: str) -> list[str]:
    """Return ``pkg-config`` output as a list of shell-tokenised args."""
    out = subprocess.check_output(["pkg-config", *args], text=True)
    return shlex.split(out)


def _split_flags(flags: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Bucket pkg-config output into setuptools' Extension kwargs.

    Returns ``(include_dirs, library_dirs, libraries, other)`` where
    *other* are remaining ``-D…`` and similar that go to
    ``extra_compile_args``.
    """
    includes, libdirs, libs, other = [], [], [], []
    for f in flags:
        if f.startswith("-I"):
            includes.append(f[2:])
        elif f.startswith("-L"):
            libdirs.append(f[2:])
        elif f.startswith("-l"):
            libs.append(f[2:])
        else:
            other.append(f)
    return includes, libdirs, libs, other


# Distros disagree on the package name:
# * Arch / Fedora / recent Debian-Ubuntu ship the upstream PulseAudio
#   fork as ``webrtc-audio-processing-1``.
# * Older Debian-Ubuntu ship the legacy fork as
#   ``webrtc-audio-processing``.
# * Some niche distros use ``webrtc_audio_processing``.
#
# Probe in that order; bail with an actionable error message if none
# resolves so users know exactly which package they're missing.
_CANDIDATE_PKGS = (
    "webrtc-audio-processing-1",
    "webrtc-audio-processing",
    "webrtc_audio_processing",
)


def _require_native_extension() -> bool:
    value = os.environ.get("AXIO_REALTIME_CHAT_REQUIRE_WEBRTC_APM", "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _missing_pkg_message() -> str:
    return (
        "webrtc-audio-processing not found via pkg-config.  Install one of:\n"
        "  Arch:           pacman -S webrtc-audio-processing\n"
        "  Debian/Ubuntu:  apt install libwebrtc-audio-processing-dev "
        "(or libwebrtc-audio-processing1-dev)\n"
        "  Fedora:         dnf install webrtc-audio-processing-devel\n"
        "Tried pkg-config names: " + ", ".join(_CANDIDATE_PKGS)
    )


def _resolve_pkg() -> str | None:
    for name in _CANDIDATE_PKGS:
        try:
            subprocess.check_call(
                ["pkg-config", "--exists", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return name
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def _build_ext_modules() -> list[Pybind11Extension]:
    pkg = _resolve_pkg()
    if pkg is None:
        message = _missing_pkg_message()
        if _require_native_extension():
            raise SystemExit(message)
        print(f"{message}\nSkipping optional webrtc_apm extension build.", file=sys.stderr)
        return []

    cflags = _pkg_config("--cflags", pkg)
    ldflags = _pkg_config("--libs", pkg)

    inc_c, lib_c, libs_c, extra_c = _split_flags(cflags)
    inc_l, lib_l, libs_l, extra_l = _split_flags(ldflags)

    return [
        Pybind11Extension(
            "webrtc_apm",
            sources=["webrtc_apm.cpp"],
            include_dirs=[*inc_c, *inc_l],
            library_dirs=[*lib_c, *lib_l],
            libraries=[*libs_c, *libs_l],
            extra_compile_args=[*extra_c, *extra_l],
            cxx_std=17,
        )
    ]


setup(ext_modules=_build_ext_modules(), cmdclass={"build_ext": build_ext})
