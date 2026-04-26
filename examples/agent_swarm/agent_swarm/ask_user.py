from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Annotated

from axio.tool import Tool, ToolHandler
from pydantic import Field
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

PromptFn = Callable[[], str] | None


def _default_prompt() -> str:
    return input("> ").strip() or "(no answer)"


class AskUser(ToolHandler[PromptFn]):
    """Ask the user a question and return their answer.
    Use this when a decision requires human input before proceeding.
    The `question` field MUST be formatted as Markdown."""

    question: Annotated[str, Field(description="Question to ask the user, formatted as Markdown")]

    async def __call__(self, context: PromptFn) -> str:
        return await asyncio.to_thread(self._blocking, context)

    def _blocking(self, fn: PromptFn) -> str:
        console = Console(stderr=True)
        console.print(Rule("[bold yellow]Question for you[/bold yellow]"))
        console.print(Markdown(self.question))
        console.print(Rule())
        sys.stderr.flush()
        prompt = fn if fn is not None else _default_prompt
        return prompt()


def make_ask_user_tool(prompt_fn: PromptFn = None, guards: tuple = ()) -> Tool:
    """Create an ask_user tool with *prompt_fn* as its context.

    When *prompt_fn* is ``None`` the tool falls back to the default stdin prompt.
    """
    return Tool(
        name="ask_user",
        description=AskUser.__doc__ or "",
        handler=AskUser,
        context=prompt_fn,
        guards=guards,
    )
