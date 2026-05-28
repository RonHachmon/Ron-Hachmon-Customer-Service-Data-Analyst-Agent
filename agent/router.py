"""Query router for the data-analyst agent.

Classifies an incoming user message into one of three buckets before the
ReAct loop touches it:

* ``structured`` — concrete, data-driven question (counts, lists, examples).
* ``unstructured`` — open-ended summarization that still needs the data.
* ``out_of_scope`` — anything unrelated to the Bitext dataset.

The classifier itself is a small Nebius model called via
:func:`langchain_openai.ChatOpenAI.with_structured_output`, so the response is
already validated against :class:`RouterDecision` by the time it returns.

The graph node that consumes this is wired up in Phase 4; this module
intentionally exposes a plain ``classify_query`` callable so it can be tested
in isolation.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_router_model
from agent.prompts import ROUTER_SYSTEM_PROMPT

Route = Literal["structured", "unstructured", "out_of_scope"]


class RouterDecision(BaseModel):
    """Classification result from the router model."""

    route: Route = Field(
        description=(
            "How this query should be handled: 'structured' for concrete "
            "data-driven questions, 'unstructured' for open-ended summarization "
            "over the dataset, 'out_of_scope' for anything unrelated."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the choice (shown in the CLI trace)."
    )


def classify_query(query: str) -> RouterDecision:
    """Return how the router thinks ``query`` should be handled.

    The call is deterministic at the model layer (``temperature=0``) but the
    underlying API can still flake — callers in production should handle
    transient errors. For Phase 3 we let exceptions bubble up; the graph
    in Phase 4 will catch them and route to a safe fallback.
    """
    if not query.strip():
        raise ValueError("classify_query requires a non-empty query string")

    classifier = get_router_model().with_structured_output(RouterDecision)
    result = classifier.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )
    assert isinstance(result, RouterDecision)
    return result
