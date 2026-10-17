"""Account-specific ChatGPT model capabilities and limits from the API catalog."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from axio.models import Capability, ModelSpec

# These are capabilities of the implemented transport, not assertions about a model.
SUPPORTED_CAPABILITIES = frozenset({Capability.text, Capability.vision, Capability.reasoning, Capability.tool_use})


@dataclass(frozen=True, slots=True)
class ChatGPTModel(ModelSpec):
    """A server catalog entry with capabilities and limits from API metadata.

    Subscription costs are unknown, represented as NaN rather than zero. Generic
    ModelSpec token budgets remain fallbacks when no model metadata is available.
    All catalog models support text and function tools. Missing metadata can use
    an explicitly supplied ModelSpec. The API-key registry is not consulted.
    """

    input_cost: float = math.nan
    output_cost: float = math.nan
    display_name: str = ""
    visibility: str = ""
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False, compare=False)


def supported_reasoning_efforts(model: ModelSpec | None) -> tuple[str, ...]:
    """Use catalog levels when present; otherwise retain the generic fallback."""
    if model is None or Capability.reasoning not in model.capabilities:
        return ()
    if isinstance(model, ChatGPTModel) and "supported_reasoning_levels" in model.metadata:
        levels = model.metadata["supported_reasoning_levels"]
        if not isinstance(levels, list):
            return ()
        return tuple(
            dict.fromkeys(
                level["effort"]
                for level in levels
                if isinstance(level, dict) and isinstance(level.get("effort"), str) and level["effort"]
            )
        )
    return ("none", "minimal", "low", "medium", "high", "xhigh")


def model_capabilities(entry: Mapping[str, Any], fallback: ModelSpec) -> frozenset[Capability]:
    """Enable text/tools for catalog models and read optional API capabilities."""
    caps = set(fallback.capabilities & SUPPORTED_CAPABILITIES) | {Capability.text, Capability.tool_use}
    if "input_modalities" in entry:
        caps.discard(Capability.vision)
        modalities = entry["input_modalities"]
        if isinstance(modalities, list):
            if "image" in modalities:
                caps.add(Capability.vision)
    if "supported_reasoning_levels" in entry:
        caps.discard(Capability.reasoning)
        levels = entry["supported_reasoning_levels"]
        if isinstance(levels, list) and any(
            isinstance(level, dict) and isinstance(level.get("effort"), str) and level["effort"] not in {"", "none"}
            for level in levels
        ):
            caps.add(Capability.reasoning)
    return frozenset(caps)


def token_limit(entry: Mapping[str, Any], key: str, fallback: int) -> int:
    """Use a positive API token limit, with a fallback for absent fields."""
    value = entry.get(key)
    return value if type(value) is int and value > 0 else fallback


def parse_models(body: Any, known: Mapping[str, ModelSpec]) -> tuple[ChatGPTModel, ...]:
    """Validate the documented catalog shape and retain server order and hidden entries."""
    if not isinstance(body, dict) or not isinstance(body.get("models"), list):
        raise ValueError("ChatGPT model catalog must contain a models array.")
    result: list[ChatGPTModel] = []
    seen: set[str] = set()
    for entry in body["models"]:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key] for key in ("slug", "display_name", "visibility")
        ):
            raise ValueError("ChatGPT models require nonempty slug, display_name, and visibility strings.")
        slug = entry["slug"]
        if slug in seen:
            raise ValueError(f"Duplicate ChatGPT model slug: {slug}")
        seen.add(slug)
        spec = known.get(slug, ModelSpec(slug))
        result.append(
            ChatGPTModel(
                id=slug,
                display_name=entry["display_name"],
                visibility=entry["visibility"],
                capabilities=model_capabilities(entry, spec),
                context_window=token_limit(entry, "context_window", spec.context_window),
                max_output_tokens=token_limit(entry, "max_output_tokens", spec.max_output_tokens),
                metadata=MappingProxyType(deepcopy(entry)),
            )
        )
    return tuple(result)
