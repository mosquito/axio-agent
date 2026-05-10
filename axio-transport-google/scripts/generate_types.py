#!/usr/bin/env python3
"""Generate TypedDict definitions from Google Vertex AI discovery document.

API reference (discovery docs):
  https://aiplatform.googleapis.com/$discovery/rest?version=v1
  https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1

Usage:
  python scripts/generate_types.py [--version v1beta1] [--output src/axio_transport_google/_generated_types.py]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.request
from typing import Any, cast

DISCOVERY_URL = "https://aiplatform.googleapis.com/$discovery/rest?version={version}"

# Schemas relevant to the generateContent API, in dependency order.
# Keys: short name used in generated code. Values: full discovery schema id.
SCHEMA_MAP = {
    "Blob": "Blob",
    "FileData": "FileData",
    "FunctionCall": "FunctionCall",
    "FunctionResponse": "FunctionResponse",
    "FunctionResponsePart": "FunctionResponsePart",
    "ExecutableCode": "ExecutableCode",
    "CodeExecutionResult": "CodeExecutionResult",
    "VideoMetadata": "VideoMetadata",
    "Part": "Part",
    "Content": "Content",
    "FunctionDeclaration": "FunctionDeclaration",
    "Tool": "Tool",
    "FunctionCallingConfig": "FunctionCallingConfig",
    "ToolConfig": "ToolConfig",
    "ThinkingConfig": "GenerationConfigThinkingConfig",
    "GenerationConfig": "GenerationConfig",
    "SafetySetting": "SafetySetting",
    "GenerateContentRequest": "GenerateContentRequest",
    "Candidate": "Candidate",
    "UsageMetadata": "GenerateContentResponseUsageMetadata",
    "PromptFeedback": "GenerateContentResponsePromptFeedback",
    "GenerateContentResponse": "GenerateContentResponse",
}

# Mapping from discovery type strings to Python type annotations
TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "any": "Any",
}


def fetch_discovery(version: str) -> dict[str, Any]:
    url = DISCOVERY_URL.format(version=version)
    print(f"Fetching {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:
        return cast(dict[str, Any], json.loads(resp.read()))


def resolve_schema_name(full_id: str, version_prefix: str) -> str | None:
    """Extract the short schema name from the full discovery ID."""
    if full_id.startswith(version_prefix):
        return full_id[len(version_prefix) :]
    return None


def ref_to_short_name(ref: str, version_prefix: str) -> str | None:
    """Convert a $ref value to our short TypedDict name, if it's in SCHEMA_MAP."""
    short = resolve_schema_name(ref, version_prefix)
    if short is None:
        return None
    for td_name, schema_suffix in SCHEMA_MAP.items():
        if short == schema_suffix:
            return td_name
    return None


def python_type_for_property(prop: dict[str, Any], version_prefix: str, required: bool = True) -> str:
    """Convert a discovery property definition to a Python type annotation."""
    if "$ref" in prop:
        td_name = ref_to_short_name(prop["$ref"], version_prefix)
        if td_name:
            return td_name
        return "dict[str, Any]"

    prop_type = prop.get("type", "any")

    if prop_type == "array":
        items = prop.get("items", {})
        item_type = python_type_for_property(items, version_prefix)
        return f"list[{item_type}]"

    if prop_type == "object":
        additional = prop.get("additionalProperties", {})
        if additional:
            val_type = python_type_for_property(additional, version_prefix)
            return f"dict[str, {val_type}]"
        return "dict[str, Any]"

    if "enum" in prop:
        literals = [f'"{v}"' for v in prop["enum"]]
        single_line = f"Literal[{', '.join(literals)}]"
        if len(single_line) <= 80:
            return single_line
        # Multi-line Literal for long enum lists
        inner = ",\n    ".join(literals)
        return f"Literal[\n    {inner},\n]"

    return TYPE_MAP.get(prop_type, "Any")


def generate_typeddict(
    td_name: str,
    schema: dict[str, Any],
    version_prefix: str,
) -> str:
    """Generate a TypedDict class definition from a discovery schema."""
    properties = schema.get("properties", {})
    if not properties:
        return f"class {td_name}(TypedDict, total=False):\n    pass\n"

    lines = [f"class {td_name}(TypedDict, total=False):"]

    desc = schema.get("description", "")
    if desc:
        wrapped = textwrap.fill(desc, width=100, initial_indent="    ", subsequent_indent="    ")
        lines.append(f'    """{wrapped[4:]}"""')
        lines.append("")

    for prop_name, prop_def in sorted(properties.items()):
        py_type = python_type_for_property(prop_def, version_prefix)
        prop_desc = prop_def.get("description", "")
        if prop_desc:
            short_desc = prop_desc.split(".")[0].strip()
            if len(short_desc) > 90:
                short_desc = short_desc[:87] + "..."
            lines.append(f"    # {short_desc}")
        if "\n" in py_type:
            # Multi-line type (e.g. long Literal) — indent to field level
            indented = py_type.replace("\n", "\n    ")
            lines.append(f"    {prop_name}: {indented}")
        else:
            line = f"    {prop_name}: {py_type}"
            if len(line) > 119:
                # Break long single-line annotations
                lines.append(f"    {prop_name}: (")
                lines.append(f"        {py_type}")
                lines.append("    )")
            else:
                lines.append(line)

    return "\n".join(lines) + "\n"


def generate_all(discovery: dict[str, Any], version: str) -> str:
    """Generate the full _generated_types.py file content."""
    schemas = discovery.get("schemas", {})

    if version == "v1":
        version_prefix = "GoogleCloudAiplatformV1"
    else:
        version_prefix = "GoogleCloudAiplatformV1beta1"

    header = f'''\
"""TypedDict definitions for Google Vertex AI generateContent API.

Auto-generated from the Vertex AI discovery document ({version}).
Do not edit manually — regenerate with:
  python scripts/generate_types.py --version {version}

API reference (discovery docs):
  https://aiplatform.googleapis.com/$discovery/rest?version=v1
  https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict  # noqa: UP035

'''

    parts: list[str] = [header]

    # Track which schemas we actually found
    generated: list[str] = []
    missing: list[str] = []

    for td_name, schema_suffix in SCHEMA_MAP.items():
        full_id = f"{version_prefix}{schema_suffix}"
        schema = schemas.get(full_id)
        if schema is None:
            missing.append(f"# WARNING: schema {full_id!r} not found in discovery doc")
            continue
        parts.append(generate_typeddict(td_name, schema, version_prefix))
        parts.append("")
        generated.append(td_name)

    # __all__ for explicit exports
    all_items = [f'    "{n}",' for n in generated]
    all_block = "__all__ = [\n" + "\n".join(all_items) + "\n]\n"
    parts.insert(1, all_block)
    parts.insert(2, "")

    if missing:
        parts.append("\n".join(missing) + "\n")

    for m in missing:
        print(m, file=sys.stderr)

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v1beta1",
        choices=["v1", "v1beta1"],
        help="Discovery document version (default: v1beta1)",
    )
    parser.add_argument(
        "--output",
        default="src/axio_transport_google/_generated_types.py",
        help="Output file path",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Read discovery doc from file instead of fetching",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            discovery = json.load(f)
    else:
        discovery = fetch_discovery(args.version)

    content = generate_all(discovery, args.version)

    with open(args.output, "w") as f:
        f.write(content)
    print(f"Generated {args.output} ({len(SCHEMA_MAP)} schemas)", file=sys.stderr)


if __name__ == "__main__":
    main()
