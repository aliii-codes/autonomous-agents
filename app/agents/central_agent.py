"""
Master router graph — PARALLEL fan-out / fan-in workflow.

Flow:
    START → router → { customer_support | lead_generation | pitch_and_outreach }
                          (all selected agents run concurrently)
                              ↓
                          aggregate → END

- Checkpointer: Redis (AsyncRedisSaver) — from core/database.py
- The `customer_support` node delegates to the real ReAct agent in
  agents/customer_support/agent.py (MongoDB-checkpointed). Not a dummy.
- thread_id is shared with the CS agent so state stays correlated.
"""

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from core.database import get_database, get_checkpointer
from core.settings import settings
from agents.customer_support.agent import run_customer_support_turn
from agents.leads.agent import run_lead_generation_turn

logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# LLM for routing only
# ---------------------------------------------------------------------------
router_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=settings.GROQ_API_KEY,
)

AgentName = Literal["customer_support", "lead_generation", "pitch_and_outreach"]
ALL_AGENTS: List[str] = ["customer_support", "lead_generation", "pitch_and_outreach"]


# ---------------------------------------------------------------------------
# State — uses reducers for parallel-safe writes
# ---------------------------------------------------------------------------
def _merge_dicts(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Reducer for parallel branch outputs.

    IMPORTANT: This reducer is *append-style* — it merges the new branch
    output into the existing dict. If the checkpointer carries over old
    values from a previous turn, they will accumulate forever.

    We reset `agent_replies` at the start of every turn (see router_node)
    by returning an explicit empty dict — but reducers can't "subtract",
    so instead we clear at the aggregate level by only reading replies
    written THIS turn.
    """
    return {**(a or {}), **(b or {})}


class MasterState(TypedDict):
    # Chat transcript. `add_messages` reducer makes parallel AIMessage writes safe.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Turn inputs (populated before invoke)
    tenant: Dict[str, Any]
    customer_phone: str
    user_message: str

    # Router output — list of agents to run in parallel
    next_agents: List[AgentName]

    # Per-agent replies, merged from parallel branches
    agent_replies: Annotated[Dict[str, Any], _merge_dicts]

    # Final composed reply (written by aggregate node)
    final_reply: str


# ---------------------------------------------------------------------------
# Router response schema
# ---------------------------------------------------------------------------
class RouteResponse(BaseModel):
    next_agents: List[AgentName] = Field(
        description=(
            "One or more destination agents. Pick multiple ONLY when the "
            "user message legitimately spans several concerns."
        )
    )


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------
async def router_node(state: MasterState) -> Dict[str, Any]:
    system_prompt = SystemMessage(
        content="""You are the router intent node for a multi-agent autonomous system.
        Select one or more destination agents for the incoming message:

        - customer_support: user tickets, ordering issues, updates, cancellations.
        - lead_generation: collect target business info (US real estate/businesses), details, contact info, gap analysis.
        - pitch_and_outreach: generate tailored outreach pitches for identified leads.

        Rules:
        - Default to a SINGLE agent.
        - Return multiple agents ONLY if the message clearly requires more than one.
        - Return at least one agent."""
    )

    structured_llm = router_llm.with_structured_output(RouteResponse)
    payload = [system_prompt] + list(state["messages"])
    result: RouteResponse = await structured_llm.ainvoke(payload)

    agents = result.next_agents or ["customer_support"]
    logger.info(f"[router] user_msg='{state['user_message'][:80]}' → agents={agents}")

    # NOTE: We do NOT return agent_replies here — the reducer is additive,
    # so we can't "clear" it from the router. See aggregate_node for the
    # filtering fix that scopes replies to *this turn's* selected agents.
    return {"next_agents": agents}


# ---------------------------------------------------------------------------
# Agent node — customer_support (real ReAct agent, MongoDB-checkpointed)
# ---------------------------------------------------------------------------
async def customer_support_node(state: MasterState) -> Dict[str, Any]:
    tenant = state["tenant"]
    customer_phone = state["customer_phone"]
    user_message = state["user_message"]

    db = get_database()

    try:
        reply = await run_customer_support_turn(
            db=db,
            tenant=tenant,
            customer_phone=customer_phone,
            user_message=user_message,
        )
    except Exception as e:
        logger.exception(f"[master.customer_support] agent failed: {e}")
        reply = (
            "Maazrat, abhi technical masla aa gaya hai. "
            "Thori dair baad dobara koshish karein."
        )

    return {
        "agent_replies": {"customer_support": reply},
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Agent node — lead_generation (real ReAct agent, file-backed)
# ---------------------------------------------------------------------------
async def lead_generation_node(state: MasterState) -> Dict[str, Any]:
    tenant = state["tenant"]
    customer_phone = state["customer_phone"]
    user_message = state["user_message"]

    db = get_database()

    try:
        reply = await run_lead_generation_turn(
            db=db,
            tenant=tenant,
            customer_phone=customer_phone,
            user_message=user_message,
        )
    except Exception as e:
        logger.exception(f"[master.lead_generation] agent failed: {e}")
        reply = "Lead generation mein abhi masla aa gaya hai. Thori dair baad try karein."

    return {
        "agent_replies": {"lead_generation": reply},
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Agent node — pitch_and_outreach (placeholder for now)
# ---------------------------------------------------------------------------
async def pitch_and_outreach_node(state: MasterState) -> Dict[str, Any]:
    await asyncio.sleep(0)
    reply = "[pitch_and_outreach] not yet implemented."
    return {
        "agent_replies": {"pitch_and_outreach": reply},
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Fan-in / aggregate node
# ---------------------------------------------------------------------------
async def aggregate_node(state: MasterState) -> Dict[str, Any]:
    """
    Runs once after ALL parallel branches finish.

    FIX: Only consider replies from agents that were selected THIS turn.
    Because `agent_replies` uses an additive reducer and is persisted by
    the Redis checkpointer, old replies from previous turns would
    otherwise bleed into `final_reply`.
    """
    all_replies = state.get("agent_replies", {}) or {}
    selected = set(state.get("next_agents") or [])

    # Scope to only this turn's selected agents
    replies = {name: text for name, text in all_replies.items() if name in selected}

    # Fallback: if router selected something but no reply landed (crash),
    # fall back to the raw dict rather than silently returning nothing.
    if not replies and all_replies:
        replies = all_replies

    if not replies:
        final = "Ji, aap ka paigham mosool ho gaya hai."
    elif len(replies) == 1:
        # Single agent → return its reply verbatim, no bullet prefix
        final = next(iter(replies.values()))
    else:
        # Multi-agent → clean composition, no "• agent_name:" clutter
        final = "\n\n".join(replies.values())

    return {"final_reply": final}


# ---------------------------------------------------------------------------
# Conditional edges — fan out to all selected agents
# ---------------------------------------------------------------------------
def route_to_agents(state: MasterState) -> List[str]:
    """
    Returns a list of node names — LangGraph executes them in parallel.
    Unknown entries are filtered out to avoid KeyErrors.
    """
    wanted = state.get("next_agents") or ["customer_support"]
    return [a for a in wanted if a in ALL_AGENTS] or ["customer_support"]


def _build_master_graph() -> StateGraph:
    graph = StateGraph(MasterState)

    graph.add_node("router", router_node)
    graph.add_node("customer_support", customer_support_node)
    graph.add_node("lead_generation", lead_generation_node)
    graph.add_node("pitch_and_outreach", pitch_and_outreach_node)
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_to_agents,
        {
            "customer_support": "customer_support",
            "lead_generation": "lead_generation",
            "pitch_and_outreach": "pitch_and_outreach",
        },
    )

    graph.add_edge("customer_support", "aggregate")
    graph.add_edge("lead_generation", "aggregate")
    graph.add_edge("pitch_and_outreach", "aggregate")

    graph.add_edge("aggregate", END)

    return graph

_master_graph: CompiledStateGraph | None = None


def get_master_graph() -> CompiledStateGraph:
    global _master_graph
    if _master_graph is None:
        checkpointer = get_checkpointer()
        _master_graph = _build_master_graph().compile(checkpointer=checkpointer)
    return _master_graph


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def run_master_turn(
    tenant: Dict[str, Any],
    customer_phone: str,
    user_message: str,
) -> Dict[str, Any]:
    tenant_id = tenant.get("tenant_id", "default_tenant")
    thread_id = f"{tenant_id}:{customer_phone}"

    graph = get_master_graph()

    config = {
        "configurable": {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "customer_phone": customer_phone,
        },
        "run_name": f"master-turn-{customer_phone}",
        "tags": [tenant_id, "master_graph", "production", "parallel"],
        "metadata": {
            "tenant_id": tenant_id,
            "customer_phone": customer_phone,
            "thread_id": thread_id,
        },
    }

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "tenant": tenant,
            "customer_phone": customer_phone,
            "user_message": user_message,
        },
        config=config,
    )

    return result