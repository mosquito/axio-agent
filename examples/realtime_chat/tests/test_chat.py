from __future__ import annotations

from pathlib import Path

from axio import Tool

from chat import build_voice_prompt


async def _dummy_tool() -> str:
    """Return a dummy value."""
    return "ok"


def test_build_voice_prompt_includes_tools_root_and_language() -> None:
    prompt = build_voice_prompt(
        "You are concise.",
        [Tool(name="dummy", handler=_dummy_tool)],
        "ru-RU",
        fs_root=Path("/tmp/workspace"),
    )

    assert "You are concise." in prompt
    assert "dummy: Return a dummy value." in prompt
    assert "Filesystem tools are read-only and restricted to /tmp/workspace." in prompt
    assert "Always respond in Russian." in prompt
