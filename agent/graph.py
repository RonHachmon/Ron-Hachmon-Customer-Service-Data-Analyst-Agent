"""LangGraph wiring for the data-analyst agent.

Graph shape::

    START
      → router
          → decline          (out_of_scope; terminal)
          → agent ⇄ tools    (structured / unstructured ReAct loop)
              → fallback      (iteration cap hit; terminal)
              → END           (final answer)

A custom ``StateGraph`` (rather than ``create_react_agent``) so the CLI in
Phase 5 can stream every router decision, tool call, and observation.

The ``profile_update`` node (Phase 7) and the SQLite checkpointer (Phase 6)
are not wired here, but :func:`build_graph` accepts an optional
``checkpointer`` so Phase 6 is a one-line change.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.memory import read_profile, update_profile
from agent.models import get_reasoner_model
from agent.prompts import (
    DECLINE_MESSAGE,
    FALLBACK_MESSAGE,
    REASONER_SYSTEM_PROMPT,
    ROUTE_HINTS,
)
from agent.router import classify_query
from agent.tools import ALL_TOOLS

DEFAULT_MAX_ITERATIONS = 12

# Run the incremental profile update once every N substantive user turns; the
# on-close consolidation in the CLI catches anything between checkpoints.
DEFAULT_PROFILE_UPDATE_EVERY = 5


class AgentState(MessagesState):
    """Conversation state.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from
    ``MessagesState`` and adds router/loop bookkeeping.
    """

    route: str
    router_reason: str
    steps: int


_CONTEXT_MSG_TRUNCATE = 200


def _latest_human_text(state: AgentState) -> str:
    """Return the content of the most recent HumanMessage in the state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("router node ran without any HumanMessage in state")


def _recent_context(state: AgentState, max_lines: int = 6) -> str:
    """Build a short transcript of prior turns for follow-up resolution.

    Includes user messages and final assistant answers (not tool noise) from
    everything before the current message, so the router can tell that "the
    total of the last two" refers to earlier counts.
    """
    lines: list[str] = []
    for message in state["messages"][:-1]:  # exclude the current user message
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            text = str(message.content)
            if len(text) > _CONTEXT_MSG_TRUNCATE:
                text = text[:_CONTEXT_MSG_TRUNCATE] + "…"
            lines.append(f"Agent: {text}")
    return "\n".join(lines[-max_lines:])


def router_node(state: AgentState) -> dict:
    """Classify the latest user message (Phase 3) into a route.

    Passes recent conversation context so follow-ups that are meaningless in
    isolation ("what about refunds?", "total of the last two?") aren't wrongly
    classified as out_of_scope.
    """
    decision = classify_query(_latest_human_text(state), context=_recent_context(state))
    return {"route": decision.route, "router_reason": decision.reason}


def decline_node(state: AgentState) -> dict:
    """Terminal node for out-of-scope queries — fixed, LLM-free reply."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def _session_id(config: RunnableConfig) -> str:
    """Extract the session/thread id from the run config."""
    return (config or {}).get("configurable", {}).get("thread_id", "default")


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """One reasoning step: call the tool-bound reasoner on the conversation.

    Injects the user profile (if any) as a ``## User context`` block so the
    agent can answer "what do you remember about me?" and tailor responses.
    """
    route = state.get("route", "")
    system_prompt = REASONER_SYSTEM_PROMPT + ROUTE_HINTS.get(route, "")
    profile = read_profile(_session_id(config))
    if profile.strip():
        system_prompt += f"\n\n## User context (what you know about this user)\n{profile}"

    # Profile turns are answered conversationally from the User-context block —
    # binding dataset tools here invites a needless tool spiral (e.g. fetching
    # refund stats just because the user said they like refunds).
    base_model = get_reasoner_model()
    model = base_model if route == "profile" else base_model.bind_tools(ALL_TOOLS)
    response = model.invoke([SystemMessage(content=system_prompt), *state["messages"]])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def profile_update_node(state: AgentState, config: RunnableConfig) -> dict:
    """Distill the latest exchange into the persistent user profile.

    Reached only on save turns (every Nth turn, gated by ``should_continue``);
    never after decline/fallback. Uses the cheap model via
    :func:`agent.memory.update_profile`. Returns no state change — the profile
    lives in its own file, not in conversation state.
    """
    last_user = _latest_human_text(state)
    last_answer = ""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            last_answer = str(message.content)
            break
    excerpt = f"User: {last_user}\nAgent: {last_answer}"
    update_profile(_session_id(config), excerpt)
    return {}


def fallback_node(state: AgentState) -> dict:
    """Terminal node when the loop exhausts its iteration budget."""
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def _route_after_router(state: AgentState) -> str:
    """Only out-of-scope is declined; profile/structured/unstructured → agent."""
    return "decline" if state["route"] == "out_of_scope" else "agent"


def build_graph(
    checkpointer=None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    profile_update_every: int = DEFAULT_PROFILE_UPDATE_EVERY,
) -> CompiledStateGraph:
    """Compile the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent memory
            (wired in Phase 6). ``None`` keeps the graph stateless across runs.
        max_iterations: Max number of reasoner (``agent``) visits before the
            loop gives up and returns ``FALLBACK_MESSAGE``. The cap is only
            enforced when the model wants to call another tool — a finished
            text answer is never blocked.
        profile_update_every: Run the incremental profile update once every N
            substantive user turns. The CLI's on-close consolidation catches
            anything between these checkpoints.
    """

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            # Final answer. Save the profile every Nth turn; otherwise end now.
            turn_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
            if profile_update_every > 0 and turn_count % profile_update_every == 0:
                return "profile_update"
            return END
        if state.get("steps", 0) >= max_iterations:
            return "fallback"  # wanted more tools but out of budget
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("decline", decline_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("profile_update", profile_update_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", _route_after_router, {"decline": "decline", "agent": "agent"}
    )
    builder.add_edge("decline", END)
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "fallback": "fallback",
            "profile_update": "profile_update",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("profile_update", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
