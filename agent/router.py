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

Route = Literal["structured", "unstructured", "profile", "out_of_scope"]


class RouterDecision(BaseModel):
    """Classification result from the router model."""

    route: Route = Field(
        description=(
            "How this query should be handled: 'structured' for concrete "
            "data-driven questions, 'unstructured' for open-ended summarization "
            "over the dataset, 'profile' for questions about the user themselves "
            "or what the agent remembers about them, 'out_of_scope' for anything "
            "else unrelated to the dataset."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the choice (shown in the CLI trace)."
    )


def classify_query(query: str, context: str = "") -> RouterDecision:
    """Return how the router thinks ``query`` should be handled.

    Args:
        query: The latest user message to classify.
        context: Optional transcript of recent turns. Needed to resolve
            follow-ups like "what's the total of the last two?", which are
            meaningless in isolation — without context the router would wrongly
            tag them out_of_scope.

    The call is deterministic at the model layer (``temperature=0``) but the
    underlying API can still flake — callers should handle transient errors.
    The graph catches exceptions and routes to a safe fallback.
    """
    if not query.strip():
        raise ValueError("classify_query requires a non-empty query string")

    messages: list = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
    if context.strip():
        messages.append(
            SystemMessage(
                content="Recent conversation (for resolving follow-ups):\n" + context
            )
        )
    messages.append(HumanMessage(content=query))

    classifier = get_router_model().with_structured_output(RouterDecision)
    result = classifier.invoke(messages)
    assert isinstance(result, RouterDecision)
    return result
