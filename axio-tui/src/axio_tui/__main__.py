from __future__ import annotations

import shlex
import sys
from pathlib import Path

from textual_serve.server import Server

from .app import AgentApp
from .options import StartupOptions
from .paths import ConfigPaths


def main() -> None:
    from .args import Args

    args = Args().parse_args()
    args.log.configure()

    if args.serve:
        host, _, port_str = args.web.listen.partition(":")
        port = int(port_str) if port_str else 8086
        host = host or "localhost"
        server = Server(
            command=shlex.join([sys.executable, "-m", "axio_tui", *(a for a in sys.argv[1:] if a != "--serve")]),
            host=host,
            port=port,
            title="Axio Agent",
        )
        server.serve()
    else:
        app = AgentApp(
            paths=ConfigPaths.discover(Path.cwd(), args.project_root, args.additional_root),
            options=StartupOptions(
                isolated=args.isolated,
                instructions_files=tuple(args.instructions_file),
                mcp_files=tuple(args.mcp_config),
                skill_directories=tuple(args.skills_dir),
            ),
        )
        app.run()


if __name__ == "__main__":
    main()
