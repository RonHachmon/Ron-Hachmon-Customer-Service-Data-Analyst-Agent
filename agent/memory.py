"""Persistent memory for the agent.

Two separate stores, both under ``data/`` (gitignored):

* **Conversation (episodic)** — a LangGraph ``SqliteSaver`` checkpointer at
  ``data/checkpoints.sqlite``, keyed by session ``thread_id``. Survives
  restarts (unlike ``MemorySaver``). See :func:`get_checkpointer`.
* **User profile** — a distilled markdown file per session at
  ``data/profiles/<session_id>.md``, kept deliberately separate from the
  conversation history. See :func:`read_profile` / :func:`update_profile`.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.models import get_router_model
from agent.prompts import PROFILE_UPDATE_SYSTEM_PROMPT

DATA_DIR = Path("data")
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"
PROFILE_DIR = DATA_DIR / "profiles"


@contextmanager
def get_checkpointer() -> Iterator[SqliteSaver]:
    """Yield a SQLite-backed checkpointer for persistent conversation memory.

    Used as a context manager so the underlying connection is opened for the
    lifetime of the REPL and closed cleanly on exit::

        with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            ...
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        yield checkpointer


# --- User profile (distilled facts, stored separately from conversation) ---

_SAFE_KEY = re.compile(r"[^A-Za-z0-9_-]")
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def profile_path(session_id: str) -> Path:
    """Return the markdown profile path for a session, sanitizing the key.

    The session id comes from a user-supplied ``--session`` flag, so unsafe
    characters (path separators, ``..``) are replaced to prevent traversal.
    """
    safe = _SAFE_KEY.sub("_", session_id) or "default"
    return PROFILE_DIR / f"{safe}.md"


def read_profile(session_id: str) -> str:
    """Return the stored profile markdown, or an empty string if none exists."""
    path = profile_path(session_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def write_profile(session_id: str, markdown: str) -> None:
    """Persist the profile markdown for a session."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path(session_id).write_text(markdown, encoding="utf-8")


def update_profile(session_id: str, conversation_excerpt: str) -> None:
    """Distill durable facts from a conversation excerpt into the profile.

    Reads the current profile, asks the cheap router model to merge any new
    durable facts (name, recurring topics, preferences) per
    :data:`PROFILE_UPDATE_SYSTEM_PROMPT`, and writes the result back. A no-op
    if the excerpt is empty. Used by both the per-turn graph node and the
    on-close consolidation hook.
    """
    if not conversation_excerpt.strip():
        return

    current = read_profile(session_id) or "# User Profile\n- (nothing known yet)"
    user_message = (
        f"CURRENT PROFILE:\n{current}\n\n"
        f"RECENT CONVERSATION:\n{conversation_excerpt}\n\n"
        "Return the updated profile."
    )
    response = get_router_model().invoke(
        [
            SystemMessage(content=PROFILE_UPDATE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    markdown = _CODE_FENCE.sub("", str(response.content)).strip()
    if markdown:
        write_profile(session_id, markdown)
