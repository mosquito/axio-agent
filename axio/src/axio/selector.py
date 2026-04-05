from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from axio.messages import Message
from axio.tool import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(self, messages: Iterable[Message], tools: Iterable[Tool]) -> Iterable[Tool]: ...
