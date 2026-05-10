from __future__ import annotations

import asyncio
import json

from smoke import Result, get_weather, serialise


def test_get_weather_uses_fixture_data() -> None:
    result = asyncio.run(get_weather("Tokyo"))
    assert "Tokyo" in result
    assert "rain" in result


def test_serialise_result() -> None:
    payload = json.loads(serialise(Result(scenario="tool", provider="openai", audio_bytes=12)))
    assert payload["scenario"] == "tool"
    assert payload["provider"] == "openai"
    assert payload["audio_bytes"] == 12
    assert "pcm_buffer" not in payload
