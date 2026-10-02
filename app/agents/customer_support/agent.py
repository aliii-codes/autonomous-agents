import os
from dotenv import load_dotenv
import logging
import asyncio
import weakref
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from core.settings import settings
from memory.mongo_checkpointer import MongoDBCkptSaver
from memory.context_builder import build_agent_context, trigger_summarization_if_needed
from agents.customer_support.tools import (
    search_uploaded_documents,
    create_order_tool,
    get_order_status_tool,
    update_order_tool,
    cancel_order_tool,
)

logger = logging.getLogger(__name__)

load_dotenv()



model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    groq_api_key=settings.GROQ_API_KEY,
)

tools = [
    search_uploaded_documents,
    create_order_tool,
    get_order_status_tool,
    update_order_tool,
    cancel_order_tool,
]

_agent_cache: Dict[str, CompiledStateGraph] = {}


# ---------------------------------------------------------------------------
# Per-thread locks (self-cleaning, no unbounded defaultdict)
# ---------------------------------------------------------------------------
_thread_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
    weakref.WeakValueDictionary()
)
_locks_guard = asyncio.Lock()


async def _get_thread_lock(thread_id: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[thread_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Agent factory (cached per process) — MongoDB checkpointer
# ---------------------------------------------------------------------------
def get_customer_support_agent(db: Any) -> CompiledStateGraph:
    """Returns or compiles the persistent customer support agent backed by MongoDB checkpointer."""
    if "master" not in _agent_cache:
        checkpointer = MongoDBCkptSaver(db)
        _agent_cache["master"] = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
        )
    return _agent_cache["master"]


# ---------------------------------------------------------------------------
# Reply extraction helper
# ---------------------------------------------------------------------------
def _extract_latest_ai_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        if isinstance(msg.content, list):
            text = "".join(
                b.get("text", "") for b in msg.content if isinstance(b, dict)
            )
        else:
            text = str(msg.content)
        if text.strip():
            return text
    return ""


# ---------------------------------------------------------------------------
# Main turn handler
# ---------------------------------------------------------------------------
async def run_customer_support_turn(
    db: Any,
    tenant: Dict[str, Any],
    customer_phone: str,
    user_message: str,
) -> str:
    """
    Executes a turn of customer support with hierarchical memory:
    - System prompt with static config + customer profile + conversation summary
    - Recent messages (last 6 turns) passed via state
    - Automatic summarization every 6 turns or when token threshold exceeded
    """
    tenant_id = tenant.get("tenant_id", "default_tenant")
    thread_id = f"{tenant_id}:{customer_phone}"

    lock = await _get_thread_lock(thread_id)

    async with lock:
        context = await build_agent_context(tenant, customer_phone, thread_id, db)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "customer_phone": customer_phone,
            },
            "run_name": f"whatsapp-turn-{customer_phone}",
            "tags": [tenant_id, "whatsapp", "production"],
            "metadata": {
                "tenant_id": tenant_id,
                "customer_phone": customer_phone,
                "thread_id": thread_id,
            },
        }

        agent = get_customer_support_agent(db)

        current_state = await agent.aget_state(config)
        existing_messages: List[BaseMessage] = (
            current_state.values.get("messages", []) if current_state else []
        )

        clean_history = [
            m for m in existing_messages if not isinstance(m, SystemMessage)
        ]

        recent_messages = context.recent_messages

        exec_messages: List[BaseMessage] = (
            [SystemMessage(content=context.system_prompt)]
            + [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]
            + [HumanMessage(content=user_message)]
        )

        try:
            result = await agent.ainvoke({"messages": exec_messages}, config=config)
        except Exception as e:
            logger.exception(f"Agent invocation failed for thread {thread_id}: {e}")
            return (
                "Maazrat, abhi technical masla aa gaya hai. "
                "Thori dair baad dobara koshish karein."
            )

        reply_text = _extract_latest_ai_text(result.get("messages", []))

        asyncio.create_task(
            trigger_summarization_if_needed(tenant_id, customer_phone, thread_id)
        )

        return reply_text or (
            "Ji, aap ka paigham mosool ho gaya hai. "
            "Hum aap ki mazeed kya madad kar sakte hain?"
        )