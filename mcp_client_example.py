"""Minimal MCP client for the Bitext data-analyst server.

Spawns ``mcp_server.py`` over stdio, lists its tools, and calls each one. Run
with::

    python mcp_client_example.py

A call result exposes the value two ways: ``result.data`` is the deserialized
Python object (e.g. an ``int`` or ``list[dict]``) — used below — while
``result.content[0].text`` is its raw text rendering.
"""

import asyncio
import contextlib
import sys

from fastmcp import Client


def _silence_proactor_teardown() -> None:
    """Mute Windows asyncio's harmless 'unclosed transport' shutdown noise.

    On Windows the ProactorEventLoop's subprocess transports are garbage-
    collected after the loop has already closed; their ``__del__`` then fails
    building a ResourceWarning against the closed pipe and prints a scary (but
    harmless) traceback. We wrap those ``__del__``s to ignore that one error so
    the demo exits cleanly. No effect on the actual tool results above.
    """
    from asyncio.base_subprocess import BaseSubprocessTransport
    from asyncio.proactor_events import _ProactorBasePipeTransport

    for cls in (_ProactorBasePipeTransport, BaseSubprocessTransport):
        original_del = cls.__del__

        def quiet_del(self, _orig=original_del):
            with contextlib.suppress(RuntimeError, ValueError):
                _orig(self)

        cls.__del__ = quiet_del


_RULE = "-" * 72


def _oneline(text: str, limit: int = 150) -> str:
    """Collapse whitespace/newlines and truncate long text for tidy display."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _print_section(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _print_rows(rows: list[dict]) -> None:
    """Pretty-print the list[dict] returned by get_examples / search_instructions."""
    for i, row in enumerate(rows, 1):
        print(f"  [{i}] {row['category']} / {row['intent']}")
        print(f"      Q: {_oneline(row['instruction'])}")
        print(f"      A: {_oneline(row['response'])}")


async def main() -> None:
    async with Client("mcp_server.py") as client:  # spawns the server over stdio
        tools = await client.list_tools()
        _print_section("Available tools")
        for tool in tools:
            print(f"  - {tool.name}")

        # count_rows: how many rows match a filter (here, the REFUND category).
        count = await client.call_tool("count_rows", {"category": "REFUND"})
        _print_section('count_rows(category="REFUND")')
        print(f"  {count.data:,} rows")

        # get_examples: pull 2 sample rows from the SHIPPING category.
        examples = await client.call_tool("get_examples", {"category": "SHIPPING", "n": 2})
        _print_section('get_examples(category="SHIPPING", n=2)')
        _print_rows(examples.data)

        # search_instructions: fuzzy keyword search over the instruction column.
        results = await client.call_tool("search_instructions", {"query": "wrong address"})
        _print_section('search_instructions(query="wrong address")')
        _print_rows(results.data)


if __name__ == "__main__":
    if sys.platform == "win32":
        _silence_proactor_teardown()
    asyncio.run(main())
