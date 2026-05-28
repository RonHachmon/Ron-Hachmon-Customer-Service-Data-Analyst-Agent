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
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

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


class AgentState(MessagesState):
    """Conversation state.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from
    ``MessagesState`` and adds router/loop bookkeeping.
    """

    route: str
    router_reason: str
    steps: int


def _latest_human_text(state: AgentState) -> str:
    """Return the content of the most recent HumanMessage in the state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("router node ran without any HumanMessage in state")


def router_node(state: AgentState) -> dict:
    """Classify the latest user message (Phase 3) into a route."""
    decision = classify_query(_latest_human_text(state))
    return {"route": decision.route, "router_reason": decision.reason}


def decline_node(state: AgentState) -> dict:
    """Terminal node for out-of-scope queries — fixed, LLM-free reply."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def agent_node(state: AgentState) -> dict:
    """One reasoning step: call the tool-bound reasoner on the conversation."""
    system_prompt = REASONER_SYSTEM_PROMPT + ROUTE_HINTS.get(state.get("route", ""), "")
    model = get_reasoner_model().bind_tools(ALL_TOOLS)
    response = model.invoke([SystemMessage(content=system_prompt), *state["messages"]])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def fallback_node(state: AgentState) -> dict:
    """Terminal node when the loop exhausts its iteration budget."""
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def _route_after_router(state: AgentState) -> str:
    return "decline" if state["route"] == "out_of_scope" else "agent"


def build_graph(
    checkpointer=None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> CompiledStateGraph:
    """Compile the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent memory
            (wired in Phase 6). ``None`` keeps the graph stateless across runs.
        max_iterations: Max number of reasoner (``agent``) visits before the
            loop gives up and returns ``FALLBACK_MESSAGE``. The cap is only
            enforced when the model wants to call another tool — a finished
            text answer is never blocked.
    """

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END  # model produced a final answer
        if state.get("steps", 0) >= max_iterations:
            return "fallback"  # wanted more tools but out of budget
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("decline", decline_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", _route_after_router, {"decline": "decline", "agent": "agent"}
    )
    builder.add_edge("decline", END)
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "fallback": "fallback", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
