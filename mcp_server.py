"""FastMCP server exposing the Bitext data-analyst tools over MCP.

Wraps three of the same ``@tool`` functions the ReAct agent uses
(:mod:`agent.tools`), delegating through ``.invoke(...)`` so every call runs the
Pydantic validation/normalization in :mod:`agent.schemas` — e.g. a lowercase
``category='refund'`` is normalized to ``'REFUND'`` before it hits pandas.
Calling the raw functions would skip that and silently return wrong counts.

This is a pure data server: the tools are plain pandas operations with no
Nebius/LLM calls, so **no API key is required** to run it.

Run::

    python mcp_server.py            # stdio transport (default)
    fastmcp run mcp_server.py       # same, via the FastMCP CLI

Only stderr is safe to write to under stdio transport (stdout carries the MCP
protocol). The wrapped tools never print, and dataset logging/warnings go to
stderr, so the stream stays clean.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from agent import tools as agent_tools
from agent.data import get_df

mcp = FastMCP("bitext-customer-service")


@mcp.tool
def count_rows(
    category: Annotated[
        str | None,
        Field(description="Optional category filter, e.g. 'REFUND'. Case-insensitive."),
    ] = None,
    intent: Annotated[
        str | None,
        Field(
            description="Optional intent filter, e.g. 'get_refund'. snake_case, "
            "case-insensitive."
        ),
    ] = None,
) -> int:
    """Count rows matching the given filters.

    Use this for "how many X?" questions. Both filters are optional and combine
    with AND: no args returns the total dataset size, ``category='REFUND'``
    counts that category, ``intent='get_refund'`` counts that intent, and both
    together counts rows matching both.
    """
    return agent_tools.count_rows.invoke({"category": category, "intent": intent})


@mcp.tool
def get_examples(
    category: Annotated[
        str | None, Field(description="Optional category filter, e.g. 'SHIPPING'.")
    ] = None,
    intent: Annotated[
        str | None,
        Field(description="Optional intent filter, e.g. 'change_shipping_address'."),
    ] = None,
    n: Annotated[int, Field(description="Number of example rows to return (1-20).")] = 3,
    offset: Annotated[
        int,
        Field(
            description="Number of matching rows to skip before returning. Use for "
            "'show me N more' follow-ups (e.g. n=3, offset=3 for the next page)."
        ),
    ] = 0,
) -> list[dict]:
    """Return up to ``n`` example rows from the dataset, skipping ``offset`` rows.

    Use this to "show examples of ...". Filters combine with AND. Each row is a
    dict with ``category``, ``intent``, ``instruction`` (what the user said) and
    ``response`` (what the agent replied). Rows are selected deterministically in
    order, so results are stable across calls.
    """
    return agent_tools.get_examples.invoke(
        {"category": category, "intent": intent, "n": n, "offset": offset}
    )


@mcp.tool
def search_instructions(
    query: Annotated[
        str,
        Field(
            description="Keyword or short phrase to substring-match (case-insensitive) "
            "against the instruction column, e.g. 'money back', 'wrong address'."
        ),
    ],
    n: Annotated[int, Field(description="Maximum number of matches to return (1-20).")] = 5,
) -> list[dict]:
    """Find rows whose instruction contains ``query`` (case-insensitive substring).

    Use this for fuzzy / concept-based lookups where the user describes what they
    want rather than naming a category or intent. Each match is a dict with
    ``category``, ``intent``, ``instruction`` and ``response``. For exact
    category / intent lookups, use ``get_examples`` instead.
    """
    return agent_tools.search_instructions.invoke({"query": query, "n": n})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    get_df()  # warm the lru_cache so the first client call isn't slow
    mcp.run()
