"""Tests for EmbeddingToolSelector (moved from axio/tests/test_selector.py)."""

from __future__ import annotations

import math

import pytest
from axio.blocks import TextBlock, ToolResultBlock
from axio.messages import Message
from axio.selector import ToolSelector
from axio.tool import Tool, ToolHandler

from axio_tui_rag.selector import (
    EmbeddingToolSelector,
    _cosine_similarity,
    _extract_last_user_text,
)


class DummyHandler(ToolHandler):
    async def __call__(self) -> str:
        return "ok"


def _make_tool(name: str, description: str = "") -> Tool:
    return Tool(name=name, description=description or f"Tool {name}", handler=DummyHandler)


class StubEmbeddingTransport:
    """Returns deterministic vectors: hash-based unit vectors for reproducibility."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._vector_for(t) for t in texts]

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        """Generate a deterministic 8-dim vector from text hash."""
        h = hash(text) & 0xFFFFFFFF
        raw = [float((h >> (i * 4)) & 0xF) - 7.5 for i in range(8)]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0:
            return [0.0] * 8
        return [x / norm for x in raw]


def test_protocol_conformance() -> None:
    selector = EmbeddingToolSelector(transport=StubEmbeddingTransport())
    assert isinstance(selector, ToolSelector)


def test_classvars() -> None:
    assert EmbeddingToolSelector.label == "Embedding Similarity"
    assert "cosine" in EmbeddingToolSelector.description.lower()


async def test_returns_all_when_few_tools() -> None:
    transport = StubEmbeddingTransport()
    selector = EmbeddingToolSelector(transport=transport, top_k=5)
    tools = [_make_tool(f"t{i}") for i in range(3)]
    messages = [Message(role="user", content=[TextBlock(text="hello")])]

    result = await selector.select(messages, tools)
    assert result == tools


async def test_selects_top_k() -> None:
    transport = StubEmbeddingTransport()
    selector = EmbeddingToolSelector(transport=transport, top_k=3)
    tools = [_make_tool(f"tool_{i}", f"description {i}") for i in range(6)]
    messages = [Message(role="user", content=[TextBlock(text="description 0")])]

    result = await selector.select(messages, tools)
    assert len(result) == 3
    # All returned tools should be from the original list
    for t in result:
        assert t in tools


async def test_pinned_always_included() -> None:
    transport = StubEmbeddingTransport()
    selector = EmbeddingToolSelector(
        transport=transport,
        top_k=2,
        pinned=frozenset({"pinned_tool"}),
    )
    tools = [
        _make_tool("pinned_tool", "totally unrelated description xyz"),
        _make_tool("tool_a", "alpha"),
        _make_tool("tool_b", "beta"),
        _make_tool("tool_c", "gamma"),
    ]
    messages = [Message(role="user", content=[TextBlock(text="alpha beta")])]

    result = await selector.select(messages, tools)
    assert len(result) == 2
    names = {t.name for t in result}
    assert "pinned_tool" in names


async def test_returns_all_on_tool_result_iteration() -> None:
    transport = StubEmbeddingTransport()
    selector = EmbeddingToolSelector(transport=transport, top_k=2)
    tools = [_make_tool(f"t{i}") for i in range(6)]
    messages = [
        Message(role="user", content=[TextBlock(text="hello")]),
        Message(role="assistant", content=[TextBlock(text="running tool")]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="result")]),
    ]

    result = await selector.select(messages, tools)
    assert result == tools


def test_cosine_similarity_identical() -> None:
    a = [1.0, 0.0, 0.0]
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector() -> None:
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


async def test_caches_embeddings() -> None:
    transport = StubEmbeddingTransport()
    selector = EmbeddingToolSelector(transport=transport, top_k=2)
    tools = [_make_tool(f"t{i}", f"desc {i}") for i in range(4)]
    messages = [Message(role="user", content=[TextBlock(text="query")])]

    await selector.select(messages, tools)
    first_count = transport.call_count

    await selector.select(messages, tools)
    # Only the query embedding should be called again, not tool descriptions
    assert transport.call_count == first_count + 1


def test_extract_last_user_text_with_text() -> None:
    messages = [Message(role="user", content=[TextBlock(text="hello world")])]
    assert _extract_last_user_text(messages) == "hello world"


def test_extract_last_user_text_tool_result() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="initial")]),
        Message(role="assistant", content=[TextBlock(text="response")]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="ok")]),
    ]
    assert _extract_last_user_text(messages) is None


def test_extract_last_user_text_empty() -> None:
    assert _extract_last_user_text([]) is None


def test_extract_last_user_text_multiple_text_blocks() -> None:
    messages = [Message(role="user", content=[TextBlock(text="hello"), TextBlock(text="world")])]
    assert _extract_last_user_text(messages) == "hello world"
