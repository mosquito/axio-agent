# Best Practices

Guidelines for building reliable, maintainable Axio applications.

## Tool Handlers

### Keep handlers focused

Each tool should do one thing well. If you find yourself writing "and also...", split into multiple tools.

<!-- name: test_focused_tools -->
```python
from typing import Any
from axio.tool import ToolHandler


class FetchUrl(ToolHandler[Any]):
    url: str
    async def __call__(self, context: Any) -> str: ...


class ParseJson(ToolHandler[Any]):
    data: str
    async def __call__(self, context: Any) -> dict: ...
```

### Use descriptive names and docstrings

The LLM uses these to decide when to call your tool.

<!-- name: test_descriptive_docstrings -->
```python
from typing import Any
from axio.tool import ToolHandler


class GeoLocate(ToolHandler[Any]):
    """Get geographic location from IP address using ip-api.com.

    Returns city, country, and coordinates as JSON.
    """
    ip: str = "auto"

    async def __call__(self, context: Any) -> str:
        return '{"city": "NYC", "country": "US"}'
```

### Validate inputs with Pydantic

Use Pydantic's built-in validation:

<!-- name: test_pydantic_validation -->
```python
from typing import Any
from pydantic import Field, field_validator
from axio.tool import ToolHandler


class FetchUrl(ToolHandler[Any]):
    url: str = Field(description="HTTP or HTTPS URL")
    timeout: int = Field(default=10, ge=1, le=60)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
```

### Return structured data when appropriate

Don't force everything to string — return dicts for structured results:

<!-- name: test_structured_result -->
```python
from typing import Any
from axio.tool import ToolHandler


class Calculate(ToolHandler[Any]):
    """Perform calculations."""
    a: float
    b: float
    operation: str

    async def __call__(self, context: Any) -> dict:
        ops = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b if b != 0 else 0,
        }
        result = ops[self.operation](self.a, self.b)
        return {"result": result, "operation": self.operation}
```

## Error Handling

### Raise HandlerError for expected failures

<!-- name: test_handler_error -->
```python
from typing import Any
from pathlib import Path
from axio.tool import ToolHandler
from axio.exceptions import HandlerError


class ReadFile(ToolHandler[Any]):
    path: str

    async def __call__(self, context: Any) -> str:
        path = Path(self.path)
        if not path.exists():
            raise HandlerError(f"File not found: {self.path}")
        return path.read_text()
```

### Use guards for validation

Move input validation to guards to keep handlers clean:

<!-- name: test_guard_validation -->
```python
from typing import Any
from axio.permission import PermissionGuard


class SanitizeInput(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        for field, value in handler.model_dump().items():
            if isinstance(value, str):
                setattr(handler, field, value.replace("<script>", ""))
        return handler
```

## Testing

### Test tools in isolation

<!-- name: test_isolated_tool -->
```python
from typing import Any
from axio.tool import ToolHandler


class FetchUrl(ToolHandler[Any]):
    url: str
    async def __call__(self, context: Any) -> str:
        return f"Fetched: {self.url}"
```

### Use StubTransport for agent tests

<!-- name: test_stub_transport_agent -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_tool_use_response, make_text_response


async def test_agent_with_tool():
    transport = StubTransport([
        make_tool_use_response("fetch", tool_input={"url": "..."}),
        make_text_response("Done"),
    ])
    agent = Agent(tools=[], transport=transport)
    result = await agent.run("Fetch example.com", MemoryContextStore())
    assert "Done" in result
```

### Test guards separately

<!-- name: test_guard_testing_separate -->
```python
import pytest
from typing import Any
from axio.tool import ToolHandler
from axio.permission import PermissionGuard
from axio.exceptions import GuardError


class WordCount(ToolHandler[Any]):
    text: str
    async def __call__(self, context: Any) -> str:
        return str(len(self.text.split()))


class MaxLengthGuard(PermissionGuard):
    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check(self, handler: Any) -> Any:
        for name, value in handler.model_dump().items():
            if isinstance(value, str) and len(value) > self.max_length:
                raise GuardError(f"Field '{name}' exceeds {self.max_length}")
        return handler


async def test_max_length_guard_allows():
    guard = MaxLengthGuard(max_length=100)
    handler = WordCount(text="short")
    result = await guard.check(handler)
    assert result is handler


async def test_max_length_guard_denies():
    guard = MaxLengthGuard(max_length=5)
    handler = WordCount(text="this is way too long")
    with pytest.raises(GuardError):
        await guard.check(handler)
```

## Configuration

### Use environment variables for secrets

<!-- name: test_env_variables -->
```python
import os
from dataclasses import dataclass, field


@dataclass
class MyTransport:
    api_key: str = field(default_factory=lambda: os.environ.get("MY_API_KEY", ""))
```

### Separate config from code

For complex applications, load configuration from files:

<!-- name: test_pydantic_settings -->
```python
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    database_url: str
    openai_api_key: str
    default_model: str = "gpt-4"
```

## Performance

### Limit concurrency on expensive tools

<!-- name: test_concurrency_limit -->
```python
from typing import Any
from axio.tool import Tool, ToolHandler


class SlowApiHandler(ToolHandler[Any]):
    async def __call__(self, context: Any) -> str:
        return "done"


tool = Tool(
    name="slow_api_call",
    description="Slow API calls",
    handler=SlowApiHandler,
    concurrency=2,
)

assert tool.concurrency == 2
```

### Reuse context stores

Don't create a new context store for each request:

<!-- name: test_reuse_context -->
```python
from axio.context import MemoryContextStore


# Bad: new store each time
async def handle_request_bad(msg: str) -> None:
    context = MemoryContextStore()


# Good: reuse store
context = MemoryContextStore()


async def handle_request_good(msg: str) -> None:
    pass
```

## Security

### Always use guards for sensitive operations

<!-- name: test_sensitive_tool_guards -->
```python
from typing import Any
from axio.tool import Tool, ToolHandler
from axio.permission import PermissionGuard


class RunSql(ToolHandler[Any]):
    query: str
    async def __call__(self, context: Any) -> str:
        return "result"


class ApiKeyGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        return handler


class RateLimitGuard(PermissionGuard):
    def __init__(self, max_per_minute: int = 10) -> None:
        self.max_per_minute = max_per_minute

    async def check(self, handler: Any) -> Any:
        return handler


tool = Tool(
    name="exec_sql",
    description="Execute SQL query",
    handler=RunSql,
    guards=(
        ApiKeyGuard(),
        RateLimitGuard(max_per_minute=10),
    ),
)

assert len(tool.guards) == 2
```

### Validate file path guards

If your tool accesses files, validate paths:

<!-- name: test_path_guard -->
```python
from pathlib import Path
from typing import Any
from axio.permission import PermissionGuard
from axio.exceptions import GuardError


class PathGuard(PermissionGuard):
    allowed_dirs: tuple[str, ...] = ("/tmp",)

    async def check(self, handler: Any) -> Any:
        path = Path(handler.path).resolve()
        allowed = [Path(d).resolve() for d in self.allowed_dirs]
        if not any(path.is_relative_to(a) for a in allowed):
            raise GuardError(f"Path not allowed: {handler.path}")
        return handler
```

## Code Organization

### Use type hints everywhere

Axio uses strict typing. Type hints help catch errors early:

<!-- name: test_type_hints -->
```python
from typing import Any
from axio.tool import ToolHandler


class MyTool(ToolHandler[Any]):
    query: str
    limit: int = 10

    async def __call__(self, context: Any) -> list[dict[str, str]]:
        return [{"id": 1, "name": "test"}]
```

## Production Deployment

### Use SQLite for production

MemoryContextStore loses data on shutdown. Use SQLiteContextStore for persistence:

```python
from axio_context_sqlite import connect, SQLiteContextStore

conn = await connect("production.db")
context = SQLiteContextStore(conn, session_id=user_session_id)
```

### Monitor token usage

Track usage from `IterationEnd` events:

```python
from axio.events import IterationEnd

async for event in agent.run_stream(msg, context):
    if isinstance(event, IterationEnd):
        print(f"Tokens: {event.usage}")
```