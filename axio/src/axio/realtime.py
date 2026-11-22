"""Duplex agent with concurrent tool execution and grouped result delivery."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from .blocks import ContentBlock, TextBlock, ToolUseBlock
from .events import (
    AudioOutputDelta,
    Error,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
    TurnStarted,
)
from .exceptions import StreamError
from .tool import Tool
from .transport import (
    RealtimePlaybackControl,
    RealtimeSession,
    RealtimeToolBatch,
    RealtimeToolResponse,
    RealtimeTransport,
)
from .types import StopReason

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingRealtimeTool:
    """Streamed arguments for one call, scoped to its provider response."""

    name: str
    response_id: str | None
    json_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RealtimeToolCall:
    """A JSON object or an argument error that must not run a handler."""

    block: ToolUseBlock
    error: str | None = None


@dataclass(slots=True)
class RealtimeAgent:
    """Drive a duplex session without blocking audio on tool execution.

    Tools in one response run concurrently through Tool.__call__. Results
    appear as ToolResult events and are delivered as one group where the
    session supports RealtimeToolBatch. Other sessions retain individual
    result delivery. Speech start invalidates pending/running tools;
    interrupt() also cancels provider generation once per turn.

    One consumer owns events(). Final transcripts remain observable after
    interruption; stale audio/caption/tool deltas with known IDs are suppressed.
    """

    system: str
    transport: RealtimeTransport
    tools: list[Tool[Any]] = field(default_factory=list)
    voice: str | None = None
    input_audio_format: str = "audio/pcm;rate=16000"
    output_audio_format: str = "audio/pcm;rate=24000"
    raise_on_error: bool = True
    close_timeout: float = field(default=5.0, kw_only=True)
    """Bound for shutdown, interruption cleanup and tool drain at stream EOF."""

    _session: RealtimeSession | None = field(default=None, init=False, repr=False)
    _pending: dict[str, PendingRealtimeTool] = field(default_factory=dict, init=False, repr=False)
    _tool_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _local_events: asyncio.Queue[StreamEvent] = field(
        default_factory=lambda: asyncio.Queue(64), init=False, repr=False
    )
    _epoch: int = field(default=0, init=False, repr=False)
    _response_id: str | None = field(default=None, init=False, repr=False)
    _muted_responses: dict[str, None] = field(default_factory=dict, init=False, repr=False)
    _interrupted: bool = field(default=False, init=False, repr=False)
    _interrupt_sent: bool = field(default=False, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _consuming: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.close_timeout) or self.close_timeout <= 0:
            raise ValueError("close_timeout must be positive and finite")

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is not None or self._consuming or self._tool_tasks:
            raise RuntimeError("Session already connected or still shutting down")
        if self._close_task is not None and not self._close_task.done():
            raise RuntimeError("Session is still shutting down")
        self._session = await self.transport.connect(
            system=self.system,
            tools=self.tools,
            voice=self.voice,
            input_audio_format=self.input_audio_format,
            output_audio_format=self.output_audio_format,
        )
        self._closing = False
        self._close_task = None
        self._epoch += 1
        self._response_id = None
        self._muted_responses.clear()
        self._interrupted = self._interrupt_sent = False
        self._local_events = asyncio.Queue(64)

    def invalidate_tools(self) -> None:
        """Invalidate before cancellation awaits, preventing stale result sends."""
        if not self._interrupted:
            self._epoch += 1
            self._interrupted = True
        if self._response_id is not None:
            self._muted_responses[self._response_id] = None
            if len(self._muted_responses) > 256:
                self._muted_responses.pop(next(iter(self._muted_responses)))
        self._pending.clear()
        for task in tuple(self._tool_tasks):
            task.cancel()

    async def settle_tasks(self, tasks: set[asyncio.Task[Any]], *, ignore_errors: bool = False) -> None:
        """Wait with a hard bound even if a handler suppresses cancellation."""
        if not tasks:
            return
        try:
            done, pending = await asyncio.wait(tasks, timeout=self.close_timeout)
        except BaseException:
            for task in tasks:
                task.add_done_callback(self.observe_task)
                task.cancel()
            raise
        for task in pending:
            task.add_done_callback(self.observe_task)
            task.cancel()
        if pending:
            raise TimeoutError("Realtime tasks did not stop before close_timeout")
        for task in done:
            if not task.cancelled() and not ignore_errors:
                task.result()
            elif not task.cancelled():
                task.exception()

    async def shutdown(self, session: RealtimeSession) -> None:
        # Close concurrently: an uncooperative tool cannot prevent socket cleanup.
        socket_close = asyncio.create_task(session.close(), name="realtime-session-close")
        socket_close.add_done_callback(self.observe_task)
        await self.settle_tasks({socket_close, *self._tool_tasks})

    async def close(self) -> None:
        """Cancel tools and close within close_timeout; cancelled callers leave
        shielded cleanup running. Uncooperative handlers cause TimeoutError and
        cannot send results to this or a subsequently connected session.
        """
        if self._close_task is None:
            if self._session is None:
                return
            self._closing = True
            session, self._session = self._session, None
            self.invalidate_tools()
            self._close_task = asyncio.create_task(self.shutdown(session), name="realtime-agent-close")
            self._close_task.add_done_callback(self.observe_task)
        await asyncio.shield(self._close_task)

    @staticmethod
    def observe_task(task: asyncio.Task[Any]) -> None:
        """Retrieve failures even when the caller cancelled its cleanup await."""
        if not task.cancelled():
            task.exception()

    @property
    def session(self) -> RealtimeSession:
        if self._session is None or self._closing:
            raise RuntimeError("Not connected. Call connect() or use 'async with'.")
        return self._session

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        await self.session.send(content)

    async def commit(self) -> None:
        await self.session.commit()

    async def truncate(self, item_id: str, audio_end_ms: int, *, content_index: int = 0) -> None:
        """Reconcile played audio; does not cancel generation or clear playback."""
        session = self.session
        if not isinstance(session, RealtimePlaybackControl):
            raise NotImplementedError("This realtime session does not support playback truncation")
        await session.truncate(item_id, audio_end_ms, content_index=content_index)

    async def interrupt(self) -> None:
        session = self.session
        self.invalidate_tools()
        tasks: set[asyncio.Task[Any]] = set(self._tool_tasks)
        if not self._interrupt_sent:
            self._interrupt_sent = True
            cancel = asyncio.create_task(session.interrupt(), name="realtime-session-interrupt")
            cancel.add_done_callback(self.observe_task)
            tasks.add(cancel)
            try:
                await self.settle_tasks(tasks)
            except BaseException:
                self._interrupt_sent = False
                raise
        else:
            await self.settle_tasks(tasks)

    def finalize_pending_tools(self, response_id: str | None) -> list[RealtimeToolCall]:
        calls: list[RealtimeToolCall] = []
        for tid, info in tuple(self._pending.items()):
            if response_id is not None and info.response_id is not None and response_id != info.response_id:
                continue
            error = None
            try:
                inp = json.loads("".join(info.json_parts) or "{}")
                if not isinstance(inp, dict):
                    raise ValueError("Arguments must be an object")
            except (ValueError, TypeError):
                inp = {}
                error = "Invalid tool arguments: expected a complete JSON object"
            calls.append(RealtimeToolCall(ToolUseBlock(id=tid, name=info.name, input=inp), error))
            self._pending.pop(tid)
        return calls

    async def execute_tool(self, call: RealtimeToolCall) -> tuple[RealtimeToolResponse, bool]:
        block = call.block
        tool = next((tool for tool in self.tools if tool.name == block.name), None)
        if call.error is not None:
            return RealtimeToolResponse(block.id, block.name, call.error), True
        if tool is None:
            return RealtimeToolResponse(block.id, block.name, f"Unknown tool: {block.name}"), True
        try:
            result = await tool(**block.input)
            if isinstance(result, str):
                content: str | list[ContentBlock] = result
            elif isinstance(result, list) and all(isinstance(block, ContentBlock) for block in result):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False)
            return RealtimeToolResponse(block.id, block.name, content), False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Realtime tool %s failed (%s)", block.name, type(exc).__name__)
            return RealtimeToolResponse(block.id, block.name, f"error: {exc}"), True

    async def dispatch_tools(
        self, calls: list[RealtimeToolCall], session: RealtimeSession, epoch: int, response_id: str | None
    ) -> None:
        async def execute(call: RealtimeToolCall) -> RealtimeToolResponse:
            response, is_error = await self.execute_tool(call)
            if self._session is session and self._epoch == epoch and not self._closing:
                text = (
                    response.content
                    if isinstance(response.content, str)
                    else "".join(block.text for block in response.content if isinstance(block, TextBlock))
                )
                await self._local_events.put(
                    ToolResult(
                        tool_use_id=response.tool_use_id,
                        name=response.name,
                        is_error=is_error,
                        content=text,
                        input=call.block.input,
                        response_id=response_id,
                    )
                )
            return response

        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(execute(call), name=f"realtime-tool-{call.block.name}") for call in calls]
            if self._session is not session or self._epoch != epoch or self._closing:
                return
            results = [task.result() for task in tasks]
            if isinstance(session, RealtimeToolBatch):
                await session.send_tool_results(results)
            else:
                for result in results:
                    await session.send_tool_result(result.tool_use_id, result.name, result.content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to deliver tool result group (%s): %s",
                ", ".join(call.block.id for call in calls),
                type(exc).__name__,
            )
            if self._session is session and self._epoch == epoch and not self._closing:
                error = StreamError("Failed to deliver realtime tool results")
                error.__cause__ = exc
                await self._local_events.put(Error(exception=error))

    def handle_event(self, event: StreamEvent, session: RealtimeSession) -> bool:
        """Apply lifecycle changes before exposing an event to the caller."""
        rid = getattr(event, "response_id", None)
        stale = rid is not None and rid in self._muted_responses
        if stale and isinstance(event, (AudioOutputDelta, TextDelta, TranscriptDelta, ToolUseStart, ToolInputDelta)):
            return False
        match event:
            case TurnStarted(response_id=rid):
                if stale:
                    return False
                if self._tool_tasks or self._pending:
                    self.invalidate_tools()
                self._response_id = rid
                self._interrupted = self._interrupt_sent = False
            case SpeechStarted():
                self.invalidate_tools()
            case SpeechStopped():
                self._interrupted = self._interrupt_sent = False
            case ToolUseStart(tool_use_id=tid, name=name):
                if rid is not None:
                    self._response_id = rid
                self._pending[tid] = PendingRealtimeTool(name, rid or self._response_id)
            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                if tid in self._pending:
                    self._pending[tid].json_parts.append(pj)
            case TurnComplete(stop_reason=sr) if not stale:
                if sr is StopReason.tool_use and self._pending:
                    task = asyncio.create_task(
                        self.dispatch_tools(
                            self.finalize_pending_tools(rid), session, self._epoch, rid or self._response_id
                        ),
                        name="realtime-tool-group",
                    )
                    self._tool_tasks.add(task)
                    task.add_done_callback(self._tool_tasks.discard)
                    task.add_done_callback(self.observe_task)
                elif sr is not StopReason.tool_use:
                    self._pending.clear()
        return True

    @staticmethod
    async def next_event(source: AsyncIterator[StreamEvent]) -> StreamEvent:
        return await anext(source)

    @staticmethod
    async def close_source(source: AsyncIterator[StreamEvent]) -> None:
        if isinstance(source, AsyncGenerator):
            await source.aclose()

    async def events(self) -> AsyncGenerator[StreamEvent]:
        """Merge provider events and local tool results with bounded buffering."""
        session = self.session
        if self._consuming:
            raise RuntimeError("Only one consumer can read realtime events")
        self._consuming = True
        source = session.events()
        provider: asyncio.Task[StreamEvent] | None = asyncio.create_task(self.next_event(source))
        local: asyncio.Task[StreamEvent] = asyncio.create_task(self._local_events.get())
        eof_deadline: float | None = None
        try:
            while True:
                if provider is None and not self._tool_tasks and self._local_events.empty() and not local.done():
                    break
                waiting: set[asyncio.Task[Any]] = {local, *self._tool_tasks}
                if provider is not None:
                    waiting.add(provider)
                timeout = None if eof_deadline is None else max(0.0, eof_deadline - asyncio.get_running_loop().time())
                done, _ = await asyncio.wait(waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("Realtime tools did not finish after stream EOF")
                if provider is not None and provider in done:
                    try:
                        event = provider.result()
                    except StopAsyncIteration:
                        provider = None
                        eof_deadline = asyncio.get_running_loop().time() + self.close_timeout
                    else:
                        provider = asyncio.create_task(self.next_event(source))
                        if self.handle_event(event, session):
                            if isinstance(event, Error) and self.raise_on_error:
                                raise event.exception
                            yield event
                if local in done:
                    event = local.result()
                    local = asyncio.create_task(self._local_events.get())
                    if isinstance(event, Error) and self.raise_on_error:
                        raise event.exception
                    yield event
        finally:
            self.invalidate_tools()
            readers: set[asyncio.Task[Any]] = {local}
            if provider is not None:
                readers.add(provider)
            for reader in readers:
                reader.cancel()
            try:
                await self.settle_tasks({*readers, *self._tool_tasks}, ignore_errors=True)
                source_close = asyncio.create_task(self.close_source(source))
                source_close.add_done_callback(self.observe_task)
                await self.settle_tasks({source_close}, ignore_errors=True)
            finally:
                self._consuming = False
