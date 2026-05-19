"""Nebius Token Factory model wrappers.

Two roles:

* **Router** — small, fast model used by the query-router node to classify each
  user message as ``structured`` / ``unstructured`` / ``out_of_scope``. The
  classification is the only thing that matters here, so we want low latency
  and low cost.
* **Reasoner** — larger model used inside the ReAct loop and for summarizing
  unstructured queries. Needs solid tool-calling and reasoning quality.

Both are reached over Nebius' OpenAI-compatible endpoint via
``langchain-openai``'s :class:`ChatOpenAI`. Model IDs can be overridden via
environment variables (see :data:`ROUTER_MODEL_ENV` / :data:`REASONER_MODEL_ENV`).
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

ROUTER_MODEL_ENV = "NEBIUS_ROUTER_MODEL"
REASONER_MODEL_ENV = "NEBIUS_REASONER_MODEL"

DEFAULT_ROUTER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_REASONER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _api_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Add it to your .env file or environment."
        )
    return key


@lru_cache(maxsize=1)
def get_router_model() -> ChatOpenAI:
    """Return the small, deterministic model used by the router node."""
    return ChatOpenAI(
        model=os.environ.get(ROUTER_MODEL_ENV, DEFAULT_ROUTER_MODEL),
        api_key=_api_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=0.0,
        timeout=30,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_reasoner_model() -> ChatOpenAI:
    """Return the larger model used for ReAct reasoning and summarization."""
    return ChatOpenAI(
        model=os.environ.get(REASONER_MODEL_ENV, DEFAULT_REASONER_MODEL),
        api_key=_api_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )
