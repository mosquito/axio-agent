# Model Registry

The model registry lets transports advertise which models they support and
what each model can do. This enables capability-based model selection and
cost-aware routing.

## ModelSpec

A frozen dataclass describing a single model:

<!-- name: test_model_spec -->
```python
from dataclasses import dataclass
from axio.models import Capability


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    capabilities: frozenset[Capability] = frozenset()
    max_output_tokens: int = 8192
    context_window: int = 128000
    input_cost: float = 0.0
    output_cost: float = 0.0
```

## Capability

Models declare their capabilities via a `StrEnum`:

<!-- name: test_capability_enum -->
```python
from enum import StrEnum


class Capability(StrEnum):
    text = "text"
    vision = "vision"
    reasoning = "reasoning"
    tool_use = "tool_use"
    json_mode = "json_mode"
    structured_outputs = "structured_outputs"
    embedding = "embedding"
```

## ModelRegistry

A dict-like container for `ModelSpec` values with powerful query methods:

<!-- name: test_model_registry -->
```python
from axio.models import ModelRegistry, ModelSpec, Capability

registry = ModelRegistry()
registry["gpt-4o"] = ModelSpec(
    id="gpt-4o",
    capabilities=frozenset({Capability.text, Capability.vision, Capability.tool_use}),
    context_window=128000,
    input_cost=2.50,
    output_cost=10.00,
)
```

### Query methods

All query methods return a new `ModelRegistry`, so they can be chained:

`by_prefix(prefix)`
: Filter models whose ID starts with a prefix.
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.by_prefix("gpt-4").ids()
  ```

`by_capability(*caps)`
: Keep only models that have **all** specified capabilities.
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.by_capability(Capability.vision, Capability.tool_use).ids()
  ```

`search(*q)`
: Keep models whose ID contains **all** query substrings.
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.search("gpt", "4o").ids()
  ```

`by_cost(*, output=False, desc=False)`
: Sort by input cost (default) or output cost, ascending or descending.
  <!-- name: test_model_registry -->
  ```python
  cheapest = registry.by_cost()            # cheapest input first
  priciest = registry.by_cost(desc=True)   # most expensive first
  assert cheapest.ids() == priciest.ids()[::-1]
  ```

`ids()`
: Return a plain list of model ID strings.
  <!-- name: test_model_registry -->
  ```python
  assert registry.by_capability(Capability.vision).ids() == ["gpt-4o"]
  ```

### Chaining example

<!-- name: test_model_registry -->
```python
# Find the cheapest vision-capable model with tool use
model = (
    registry
    .by_capability(Capability.vision, Capability.tool_use)
    .by_cost()
    .ids()[0]
)
assert model == "gpt-4o"
```
