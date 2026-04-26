"""Shared test fixtures for axio core."""

from __future__ import annotations

from typing import Any

import pytest

from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_echo_tool, make_stub_transport
from axio.tool import Tool


@pytest.fixture
def stub_transport() -> StubTransport:
    return make_stub_transport()


@pytest.fixture
def ephemeral_context() -> MemoryContextStore:
    return MemoryContextStore()


@pytest.fixture
def echo_tool() -> Tool[Any]:
    return make_echo_tool()
