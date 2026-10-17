"""MCP server values and connection construction, independent of configuration files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .session import MCPSession

INHERITED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP")
HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


class ConfigError(ValueError):
    """An invalid server setting, described without exposing its value."""


def _string(value: Any, key: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or "\0" in value or (not empty and not value.strip()):
        raise ConfigError(
            f"{key} must be a string without NUL" if empty else f"{key} must be a nonempty string without NUL"
        )
    return value


def _mapping(data: Mapping[str, Any], key: str) -> dict[str, str]:
    values = data.get(key, {})
    if not isinstance(values, dict):
        raise ConfigError(f"{key} must be an object of strings")
    result: dict[str, str] = {}
    header = key in {"headers", "headersFrom"}
    for name, value in values.items():
        name = _string(name, f"{key} key")
        value = _string(value, key, empty=not key.endswith("From"))
        if header and HEADER_NAME.fullmatch(name) is None:
            raise ConfigError(f"{key} contains an invalid HTTP header name")
        if not header and "=" in name:
            raise ConfigError(f"{key} contains an invalid environment key")
        if header and any(char in value for char in "\r\n"):
            raise ConfigError(f"{key} contains a line break")
        normalized = name.lower() if header else name
        if normalized in result:
            raise ConfigError(f"{key} contains a duplicate name")
        result[normalized] = value
    return result


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Describe one stdio or HTTP server using values from an MCP JSON entry.

    ``from_dict`` consumes an already parsed entry. It does not read files or
    discover roots. Applications choose names and resolve source-relative paths.
    ``connect`` constructs and opens the corresponding session. The caller owns
    the returned session and must close it.
    """

    name: str
    command: str | None = field(default=None, repr=False)
    args: list[str] = field(default_factory=list, repr=False)
    env: dict[str, str] | None = field(default=None, repr=False)
    url: str | None = field(default=None, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    timeout: float = 30.0
    scope: str = "global"
    protocol_version: str | None = None
    cwd: str | None = None
    inherit_env: bool = True
    enabled: bool = True
    env_from: dict[str, str] = field(default_factory=dict)
    headers_from: dict[str, str] = field(default_factory=dict)
    oauth: dict[str, str | int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            valid = not isinstance(self.timeout, bool) and math.isfinite(self.timeout) and self.timeout > 0
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            raise ConfigError("timeout must be a positive finite number of seconds")
        has_command, has_url = self.command is not None, self.url is not None
        if has_command and has_url or self.enabled and not (has_command or has_url):
            raise ConfigError("Exactly one of 'command' or 'url' must be set")

    @property
    def transport(self) -> str | None:
        return "stdio" if self.command is not None else "http" if self.url is not None else None

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> MCPServerConfig:
        """Validate one decoded JSON server entry without performing I/O."""
        _string(name, "server name")
        if not isinstance(data, Mapping):
            raise ConfigError("server entry must be an object")
        enabled = data.get("enabled", True)
        inherit_env = data.get("inherit_env", True)
        if not isinstance(enabled, bool) or not isinstance(inherit_env, bool):
            raise ConfigError("enabled and inherit_env must be boolean")
        if not enabled and set(data) <= {"enabled", "inherit_env"}:
            return cls(name=name, enabled=False, inherit_env=inherit_env)
        transport = data.get("type", "stdio" if "command" in data else "http" if "url" in data else None)
        if transport not in ("stdio", "http"):
            raise ConfigError("type must be stdio or http")
        common = {"type", "enabled", "timeout", "protocol_version", "inherit_env"}
        allowed = common | (
            {"command", "args", "cwd", "env", "envFrom"}
            if transport == "stdio"
            else {"url", "headers", "headersFrom", "oauth"}
        )
        if set(data) - allowed:
            raise ConfigError("unsupported fields: " + ", ".join(sorted(set(data) - allowed)))
        timeout = data.get("timeout", 30.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ConfigError("timeout must be a positive finite number")
        try:
            timeout = float(timeout)
        except OverflowError:
            raise ConfigError("timeout must be a positive finite number") from None
        protocol = _string(data["protocol_version"], "protocol_version") if "protocol_version" in data else None
        if transport == "http":
            url = _string(data.get("url"), "url")
            try:
                parsed = urlsplit(url)
                valid = (
                    parsed.scheme in {"http", "https"}
                    and bool(parsed.hostname)
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.fragment
                    and not any(char.isspace() for char in url)
                )
                _ = parsed.port
            except ValueError:
                valid = False
            if not valid:
                raise ConfigError("url must be an HTTP(S) endpoint without user information or a fragment")
            headers, refs = _mapping(data, "headers"), _mapping(data, "headersFrom")
            if headers.keys() & refs.keys():
                raise ConfigError("headers and headersFrom must not define the same header")
            oauth = data.get("oauth")
            if oauth is not None:
                if not isinstance(oauth, dict) or set(oauth) - {"client_id", "issuer", "scope", "redirect_port"}:
                    raise ConfigError("oauth must contain only client_id, issuer, scope, and redirect_port")
                oauth = dict(oauth)
                for key in ("client_id", "issuer", "scope"):
                    if key in oauth:
                        _string(oauth[key], f"oauth.{key}", empty=True)
                if oauth.get("client_id") and not oauth.get("issuer"):
                    raise ConfigError("oauth.client_id requires oauth.issuer")
                port = oauth.get("redirect_port", 0)
                if type(port) is not int or not 0 <= port <= 65535:
                    raise ConfigError("oauth.redirect_port must be an integer between 0 and 65535")
                if "authorization" in headers or "authorization" in refs:
                    raise ConfigError("OAuth cannot be combined with an Authorization header")
            return cls(
                name,
                oauth=oauth,
                url=url,
                headers=headers,
                headers_from=refs,
                timeout=timeout,
                protocol_version=protocol,
                enabled=enabled,
                inherit_env=inherit_env,
            )
        command = _string(data.get("command"), "command")
        cwd = _string(data["cwd"], "cwd") if "cwd" in data else None
        args = data.get("args", [])
        if not isinstance(args, list):
            raise ConfigError("args must be an array of strings")
        arguments = [_string(arg, "args", empty=True) for arg in args]
        env, refs = _mapping(data, "env"), _mapping(data, "envFrom")
        if env.keys() & refs.keys():
            raise ConfigError("env and envFrom must not define the same variable")
        return cls(
            name,
            command=command,
            args=arguments,
            cwd=cwd,
            env=env or None,
            env_from=refs,
            timeout=timeout,
            protocol_version=protocol,
            enabled=enabled,
            inherit_env=inherit_env,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return native JSON values, preserving environment references."""
        result: dict[str, Any] = {}
        if not self.enabled:
            result["enabled"] = False
        if self.transport is not None:
            result["type"] = self.transport
        for key in ("command", "url", "cwd", "protocol_version"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.args:
            result["args"] = list(self.args)
        for key, values in (
            ("env", self.env),
            ("envFrom", self.env_from),
            ("headers", self.headers),
            ("headersFrom", self.headers_from),
        ):
            if values:
                result[key] = dict(values)
        if self.oauth is not None:
            result["oauth"] = dict(self.oauth)
        if self.timeout != 30.0:
            result["timeout"] = self.timeout
        if not self.inherit_env:
            result["inherit_env"] = False
        return result

    @property
    def permission_identity(self) -> str:
        return hashlib.sha256(json.dumps([self.name, self.to_dict()], sort_keys=True).encode()).hexdigest()

    def resolve(self, environment: Mapping[str, str] | None = None) -> MCPServerConfig:
        """Resolve references and executable for this connection attempt."""
        if not self.enabled:
            raise ConfigError("disabled server cannot be resolved for connection")
        host = dict(os.environ if environment is None else environment)

        def references(refs: dict[str, str], *, headers: bool = False) -> dict[str, str]:
            result: dict[str, str] = {}
            for key, variable in refs.items():
                if variable not in host:
                    raise ConfigError(f"missing environment variable {variable!r}")
                value = host[variable]
                if "\0" in value or headers and any(char in value for char in "\r\n"):
                    raise ConfigError(f"invalid value from environment variable {variable!r}")
                result[key] = value
            return result

        headers = self.headers | references(self.headers_from, headers=True)
        if self.url is not None:
            return replace(self, headers=headers, headers_from={})
        assert self.command is not None
        cwd = Path(self.cwd or ".").resolve()
        if not cwd.is_dir():
            raise ConfigError("cwd is not an existing directory")
        child = host if self.inherit_env else {key: host[key] for key in INHERITED_ENV if key in host}
        child = child | (self.env or {}) | references(self.env_from)
        child["PATH"] = os.pathsep.join(
            str((cwd / (part or ".")).resolve()) for part in child.get("PATH", os.defpath).split(os.pathsep)
        )
        command = str(cwd / self.command) if "/" in self.command else self.command
        executable = shutil.which(command, path=child["PATH"])
        if executable is None:
            raise ConfigError("command was not found or is not executable")
        return replace(
            self,
            command=os.path.abspath(executable),
            cwd=str(cwd),
            env=child,
            env_from={},
            headers=headers,
            headers_from={},
            inherit_env=False,
        )

    async def connect(self, session_type: type[MCPSession] | None = None) -> MCPSession:
        """Create and open a session; close partial startup on every failure."""
        from .session import MCPSession

        session = (session_type or MCPSession)(self.resolve())
        try:
            await session.connect()
        except BaseException:
            await session.close()
            raise
        return session

    def to_settings(self) -> dict[str, str]:
        """Serialize to flat string dict for config DB persistence."""
        result: dict[str, str] = {}
        if self.command is not None:
            result["command"] = self.command
        if self.args:
            result["args"] = json.dumps(self.args)
        if self.env:
            result["env"] = json.dumps(self.env)
        if self.url is not None:
            result["url"] = self.url
        if self.headers:
            result["headers"] = json.dumps(self.headers)
        if self.oauth is not None:
            result["oauth"] = json.dumps(self.oauth)
        if self.timeout != 30.0:
            result["timeout"] = str(self.timeout)
        if self.protocol_version is not None:
            result["protocol_version"] = self.protocol_version
        if self.cwd is not None:
            result["cwd"] = self.cwd
        if not self.inherit_env:
            result["inherit_env"] = "false"
        return result

    @classmethod
    def from_settings(cls, name: str, data: dict[str, str]) -> MCPServerConfig:
        """Deserialize from flat string dict."""
        args: list[str] = json.loads(data["args"]) if "args" in data else []
        env: dict[str, str] | None = json.loads(data["env"]) if "env" in data else None
        headers: dict[str, str] = json.loads(data["headers"]) if "headers" in data else {}
        timeout = float(data["timeout"]) if "timeout" in data else 30.0
        return cls(
            name=name,
            oauth=json.loads(data["oauth"]) if "oauth" in data else None,
            command=data.get("command"),
            args=args,
            env=env,
            url=data.get("url"),
            headers=headers,
            timeout=timeout,
            protocol_version=data.get("protocol_version"),
            cwd=data.get("cwd"),
            inherit_env=data.get("inherit_env", "true") != "false",
        )
