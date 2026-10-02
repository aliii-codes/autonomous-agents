import logging
from typing import Any, Dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from core.settings import settings
from agents.leads.tools import scrape_leads

logger = logging.getLogger(__name__)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=settings.GROQ_API_KEY,
)

LEAD_GEN_SYSTEM_PROMPT = """
You are a B2B lead-generation specialist.

Rules:
- To search for leads you need BOTH a business type (query) and a
  location. If either is missing, ASK for it before calling the tool.
  Never guess.
- Only call `scrape_leads` when both are known.
- After the tool returns, present results as a short, skimmable list:
    - Business name
    - Phone
    - Website
    - Email(s) if any
  Do NOT dump raw JSON. Do NOT repeat every field.
- Mention briefly that results were saved (once, at the end).
- If the tool returns an error or 0 leads, say so honestly.
- NEVER invent phone numbers, emails, names, or websites.
- Be concise and professional.
"""

_agent_cache: Dict[str, CompiledStateGraph] = {}


def get_lead_generation_agent() -> CompiledStateGraph:
    if "master" not in _agent_cache:
        _agent_cache["master"] = create_react_agent(
            model=model,
            tools=[scrape_leads],
            prompt=LEAD_GEN_SYSTEM_PROMPT,
        )
    return _agent_cache["master"]


def _extract_latest_ai_text(messages: list[BaseMessage]) -> str:
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


async def run_lead_generation_turn(
    db: Any,
    tenant: Dict[str, Any],
    customer_phone: str,
    user_message: str,
) -> str:
    tenant_id = tenant.get("tenant_id", "default_tenant")
    thread_id = f"{tenant_id}:{customer_phone}:leads"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "customer_phone": customer_phone,
        },
        "run_name": f"lead-gen-{customer_phone}",
        "tags": [tenant_id, "lead_generation", "production"],
        "metadata": {
            "tenant_id": tenant_id,
            "customer_phone": customer_phone,
            "thread_id": thread_id,
        },
    }

    agent = get_lead_generation_agent()

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
    except Exception as e:
        logger.exception(f"[lead_generation] agent failed: {e}")
        return "Lead generation mein abhi masla aa gaya hai. Thori dair baad try karein."

    return _extract_latest_ai_text(result.get("messages", [])) or "Koi leads nahi mile."