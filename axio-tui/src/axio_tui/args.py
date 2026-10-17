from __future__ import annotations

import logging
from pathlib import Path

import argclass
from textual.logging import TextualHandler


class LogGroup(argclass.Group):
    """Logging options."""

    level: str = argclass.LogLevel  # type: ignore[assignment]
    file: str = argclass.Argument(default=None, metavar="PATH", help="log file path")

    def configure(self) -> None:
        handlers: list[logging.Handler] = [TextualHandler()]
        if self.file:
            handlers.append(logging.FileHandler(self.file, mode="a"))

        logging.basicConfig(
            level=self.level,
            format="%(asctime)s %(levelname)s[%(name)s]: %(message)s",
            handlers=handlers,
        )


class ServeGroup(argclass.Group):
    """Web serving options."""

    listen: str = argclass.Argument(
        default="localhost:8086",
        metavar="HOST:PORT",
        help="address to listen on",
    )


class Args(argclass.Parser):
    """Axio Agent TUI."""

    serve: bool = argclass.Argument(action=argclass.Actions.STORE_TRUE, help="run in web mode")
    project_root: Path | None = argclass.Argument(default=None, type=Path, help="primary workspace root")
    additional_root: list[Path] = argclass.Argument(
        default=[], action=argclass.Actions.APPEND, type=Path, help="additional workspace root"
    )
    isolated: bool = argclass.Argument(
        action=argclass.Actions.STORE_TRUE, help="load only explicit workspace configuration"
    )
    instructions_file: list[Path] = argclass.Argument(
        default=[], action=argclass.Actions.APPEND, type=Path, help="explicit instruction file"
    )
    mcp_config: list[Path] = argclass.Argument(
        default=[], action=argclass.Actions.APPEND, type=Path, help="explicit MCP configuration"
    )
    skills_dir: list[Path] = argclass.Argument(
        default=[], action=argclass.Actions.APPEND, type=Path, help="additional skills directory"
    )
    log = LogGroup(title="logging options")
    web = ServeGroup(title="web serving options")
