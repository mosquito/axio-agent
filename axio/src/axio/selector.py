from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from .messages import Message
from .tool import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(self, messages: Iterable[Message], tools: Iterable[Tool[Any]]) -> Iterable[Tool[Any]]: ...
