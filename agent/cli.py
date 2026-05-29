"""Interactive command-line interface for the data-analyst agent.

Drops into a REPL that streams the agent's reasoning — router decision, each
tool call, each observation — and then prints the final answer. Run via
``python main.py [--session <id>]``.

Conversation state is held by a SQLite-backed LangGraph checkpointer keyed on
the session's ``thread_id``, so follow-up questions ("show me 3 more") see
earlier turns and the same ``--session`` id resumes a conversation across
restarts.
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.data import summarize
from agent.graph import DEFAULT_MAX_ITERATIONS, build_graph
from agent.memory import get_checkpointer, update_profile

_OBS_TRUNCATE = 220


def _format_args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _print_trace(node: str, delta: dict) -> str | None:
    """Print the reasoning step for one node update.

    Returns the final answer text if this update produced one (from the
    ``agent``, ``decline``, or ``fallback`` node), else ``None``.
    """
    if node == "router":
        print(f"  router  -> {delta['route']}  ({delta['router_reason']})")
        return None

    if node == "agent":
        msg = delta["messages"][-1]
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"  tool    -> {call['name']}({_format_args(call['args'])})")
        # A final answer arrives as content with no tool calls.
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return str(msg.content)
        return None

    if node == "tools":
        for msg in delta["messages"]:
            if isinstance(msg, ToolMessage):
                obs = str(msg.content).replace("\n", " ")
                if len(obs) > _OBS_TRUNCATE:
                    obs = obs[:_OBS_TRUNCATE] + "…"
                print(f"  obs     <- {msg.name}: {obs}")
        return None

    if node == "profile_update":
        print("  (updating user profile…)")
        return None

    if node in ("decline", "fallback"):
        return str(delta["messages"][-1].content)

    return None


def _full_transcript(graph, config: dict) -> str:
    """Build a transcript of the whole session for on-close profile consolidation.

    Reads the persisted state from the checkpointer and keeps user messages and
    final assistant answers (dropping tool noise).
    """
    try:
        snapshot = graph.get_state(config)
    except Exception:  # noqa: BLE001 — best-effort on close
        return ""
    messages = snapshot.values.get("messages", []) if snapshot else []
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            lines.append(f"Agent: {message.content}")
    return "\n".join(lines)


def _consolidate_profile_on_close(graph, config: dict, session_id: str) -> None:
    """Holistic profile pass over the full conversation when the user exits."""
    transcript = _full_transcript(graph, config)
    if not transcript.strip():
        return
    print("Consolidating user profile…")
    try:
        update_profile(session_id, transcript)
    except Exception as exc:  # noqa: BLE001 — don't fail exit on a profile error
        print(f"  [profile consolidation skipped: {type(exc).__name__}: {exc}]")


def _print_banner(session_id: str) -> None:
    info = summarize()
    print("=" * 70)
    print("Customer Service Data Analyst Agent")
    print(
        f"Dataset: {info['row_count']:,} rows | "
        f"{len(info['categories'])} categories | {len(info['intents'])} intents"
    )
    print(f"Session: {session_id}")
    print("Ask about the Bitext customer-service dataset. Type 'exit' or Ctrl-C to quit.")
    print("=" * 70)


def run(session_id: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
    """Start the interactive REPL for the given session id."""
    # Make emoji / unicode safe on Windows consoles (cp1252 default).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": max_iterations * 2 + 6,
    }

    # SQLite-backed memory: the same --session id resumes across restarts.
    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer, max_iterations=max_iterations)
        _print_banner(session_id)

        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", ":q"}:
                print("Goodbye.")
                break

            answer_printed = False
            try:
                for chunk in graph.stream(
                    {"messages": [HumanMessage(content=user_input)], "steps": 0},
                    config,
                    stream_mode="updates",
                ):
                    for node, delta in chunk.items():
                        answer = _print_trace(node, delta)
                        # Print the answer as soon as it's produced, so the
                        # per-turn profile_update (which streams next) doesn't
                        # delay it.
                        if answer is not None and not answer_printed:
                            print(f"\nagent> {answer}")
                            answer_printed = True
            except Exception as exc:  # noqa: BLE001 — keep the REPL alive on API errors
                print(f"  [error] {type(exc).__name__}: {exc}")
                continue

            if not answer_printed:
                print("\nagent> (no answer produced)")

        # Trigger 2: holistic profile consolidation over the whole conversation.
        _consolidate_profile_on_close(graph, config, session_id)
