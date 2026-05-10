"""Root conftest — fix package shadowing under --import-mode=importlib.

Directory names like ``axio/`` shadow the installed ``axio`` package when
pytest collects from the monorepo root.  Importing the package here (before
any test conftest runs) ensures Python resolves it from the editable install.
"""

import sys

import axio  # noqa: F401
import axio_transport_anthropic  # noqa: F401
import axio_transport_openai  # noqa: F401

# Some markdown-pytest examples (e.g. axio/README.md) replace
# ``axio_transport_openai`` in ``sys.modules`` with a stub for documentation
# purposes. Pin the real module under a private alias and restore it before
# each test so cross-test pollution can't break later imports.
_REAL_AXIO_TRANSPORT_OPENAI = sys.modules["axio_transport_openai"]


def pytest_runtest_setup(item):  # noqa: D401, ARG001
    """Restore real ``axio_transport_openai`` before every test."""
    sys.modules["axio_transport_openai"] = _REAL_AXIO_TRANSPORT_OPENAI
