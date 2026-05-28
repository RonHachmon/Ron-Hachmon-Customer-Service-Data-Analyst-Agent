"""LangChain ``@tool``-decorated functions used by the ReAct agent and the
MCP server.

Each tool wraps a single pandas operation against the cached Bitext
DataFrame returned by :func:`agent.data.get_df`. Design notes:

* Inputs are validated by Pydantic schemas in :mod:`agent.schemas`. An
  invalid ``category`` or ``intent`` raises ``ValueError`` whose message
  lists the valid options — the ReAct loop surfaces that to the model so
  it can correct itself on the next step.
* Outputs are plain Python types (``int``, ``list``, ``dict``). LangChain
  stringifies them when feeding the result back into the model.
* No I/O inside tools — see the project plan for why (MCP stdio
  transport reuses these functions and a stray ``print`` would corrupt
  the protocol stream).
"""

from __future__ import annotations

from langchain_core.tools import tool

from agent.data import get_df
from agent.schemas import (
    CountRowsArgs,
    GetExamplesArgs,
    IntentDistributionArgs,
    ListIntentsArgs,
    NoArgs,
    SearchInstructionsArgs,
)


@tool(args_schema=NoArgs)
def list_categories() -> list[str]:
    """List every top-level category in the Bitext dataset.

    Use this when the user asks "what categories exist?" or before
    filtering by category if you're unsure which one to use. Returns the
    category names in alphabetical order, all uppercased.
    """
    df = get_df()
    return sorted(df["category"].unique().tolist())


@tool(args_schema=ListIntentsArgs)
def list_intents(category: str | None = None) -> list[str]:
    """List intents in the dataset, optionally restricted to one category.

    Without a ``category`` argument returns every intent in the dataset.
    With one, returns just the intents that appear under that category
    (e.g. ``category='ACCOUNT'`` → ``['create_account', 'delete_account',
    'edit_account', ...]``). Useful for discovering what specific intents
    are available before counting or sampling.
    """
    df = get_df()
    if category is not None:
        df = df[df["category"] == category]
    return sorted(df["intent"].unique().tolist())


@tool(args_schema=CountRowsArgs)
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count rows matching the given filters.

    Use this for "how many X?" questions. Both filters are optional and
    combine with AND:

    * No args → total dataset size.
    * ``category='REFUND'`` → all rows in the REFUND category.
    * ``intent='get_refund'`` → all rows with intent ``get_refund``.
    * Both → rows matching both filters.
    """
    df = get_df()
    if category is not None:
        df = df[df["category"] == category]
    if intent is not None:
        df = df[df["intent"] == intent]
    return int(len(df))


@tool(args_schema=IntentDistributionArgs)
def intent_distribution(category: str) -> dict[str, int]:
    """Return how many rows each intent has inside a single category.

    Use this when the user asks for a breakdown / distribution of intents
    within a category, e.g. "distribution of intents in ACCOUNT" → a dict
    like ``{'create_account': 1000, 'delete_account': 1000, ...}`` sorted
    descending by count.
    """
    df = get_df()
    subset = df[df["category"] == category]
    counts = subset["intent"].value_counts()
    return {intent: int(n) for intent, n in counts.items()}


@tool(args_schema=GetExamplesArgs)
def get_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> list[dict]:
    """Return up to ``n`` example rows from the dataset, skipping ``offset`` rows.

    Use this when the user asks to "show examples of ...". Filters
    combine with AND. Each returned row is a dict with:

    * ``category`` and ``intent`` — for context
    * ``instruction`` — what the user said
    * ``response`` — what the agent replied

    Rows are selected deterministically in order, so results are stable
    across calls. For "show me N more" follow-ups, keep ``n`` the same and
    advance ``offset`` (e.g. first call ``n=3, offset=0``; next call
    ``n=3, offset=3``) to get fresh, non-overlapping rows.
    """
    df = get_df()
    if category is not None:
        df = df[df["category"] == category]
    if intent is not None:
        df = df[df["intent"] == intent]
    rows = df.iloc[offset : offset + n]
    return [
        {
            "category": str(r.category),
            "intent": str(r.intent),
            "instruction": str(r.instruction),
            "response": str(r.response),
        }
        for r in rows.itertuples(index=False)
    ]


@tool(args_schema=SearchInstructionsArgs)
def search_instructions(query: str, n: int = 5) -> list[dict]:
    """Find rows whose instruction contains ``query`` (case-insensitive substring).

    Use this for **fuzzy or concept-based** lookups where the user
    describes what they want rather than naming a category or intent —
    e.g. 'money back', 'wrong address', 'forgot my password'. The match
    is a literal substring against the ``instruction`` column. For exact
    category / intent lookups, use :func:`get_examples` instead.
    """
    df = get_df()
    mask = df["instruction"].str.contains(query, case=False, regex=False, na=False)
    rows = df[mask].head(n)
    return [
        {
            "category": str(r.category),
            "intent": str(r.intent),
            "instruction": str(r.instruction),
            "response": str(r.response),
        }
        for r in rows.itertuples(index=False)
    ]


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    intent_distribution,
    get_examples,
    search_instructions,
]
