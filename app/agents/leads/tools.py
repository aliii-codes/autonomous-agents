import logging
import os
import re
from typing import Any
from datetime import datetime, timezone

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from apify_client import ApifyClientAsync

from core.settings import settings
from agents.leads.storage import save_leads_to_files

logger = logging.getLogger(__name__)

ACTOR_ID = "compass/crawler-google-places"
_client: ApifyClientAsync | None = None


def _get_client() -> ApifyClientAsync:
    global _client
    if _client is None:
        _client = ApifyClientAsync(settings.APIFY_API_TOKEN)
    return _client


def _clean_emails(emails: list | None) -> list[str]:
    if not emails:
        return []
    cleaned = []
    for e in emails:
        if isinstance(e, str):
            e = e.strip()
            if e and "@" in e and "." in e.split("@")[-1]:
                cleaned.append(e.lower())
    return list(dict.fromkeys(cleaned))


def _classify_url(url: str) -> str:
    if not url:
        return "other"
    url_lower = url.lower()
    if any(d in url_lower for d in ["facebook.com", "fb.com"]):
        return "facebook"
    if "instagram.com" in url_lower:
        return "instagram"
    if "linkedin.com" in url_lower:
        return "linkedin"
    if any(d in url_lower for d in ["twitter.com", "x.com"]):
        return "twitter"
    if "tiktok.com" in url_lower:
        return "tiktok"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    return "other"


def _strip_url_tracking(url: str) -> str:
    if not url:
        return url
    try:
        from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
        parsed = urlparse(url)
        params = [(k, v) for k, v in parse_qsl(parsed.query) if not k.startswith("utm_")]
        return urlunparse(parsed._replace(query=urlencode(params)))
    except Exception:
        return url


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _strip_url_tracking(url)


def _website_matches_name(website: str, name: str) -> bool:
    if not website or not name:
        return False
    try:
        from urllib.parse import urlparse
        domain = urlparse(website).netloc.lower().replace("www.", "")
        name_words = re.findall(r"[a-z]+", name.lower())
        for w in name_words:
            if len(w) > 3 and w in domain:
                return True
        return False
    except Exception:
        return False


def _trim_lead(lead: dict) -> dict:
    socials = {}
    raw_socials = lead.get("socials") or {}
    for k in ("facebook", "instagram", "linkedin", "twitter", "tiktok", "youtube"):
        v = raw_socials.get(k)
        if v:
            socials[k] = v

    for key in ("facebooks", "instagrams", "linkedIns", "twitters", "tiktoks", "youtubes"):
        v = lead.get(key)
        if v:
            if key == "facebooks":
                socials.setdefault("facebook", v[0] if isinstance(v, list) else v)
            elif key == "instagrams":
                socials.setdefault("instagram", v[0] if isinstance(v, list) else v)
            elif key == "linkedIns":
                socials.setdefault("linkedin", v[0] if isinstance(v, list) else v)
            elif key == "twitters":
                socials.setdefault("twitter", v[0] if isinstance(v, list) else v)
            elif key == "tiktoks":
                socials.setdefault("tiktok", v[0] if isinstance(v, list) else v)
            elif key == "youtubes":
                socials.setdefault("youtube", v[0] if isinstance(v, list) else v)

    website = lead.get("website") or lead.get("url") or ""
    website = _normalize_url(website)
    website_trusted = _website_matches_name(website, lead.get("name") or "")

    emails = _clean_emails(lead.get("emails"))

    return {
        "name": lead.get("name") or "",
        "address": lead.get("address") or lead.get("full_address") or "",
        "phone": lead.get("phone") or lead.get("phone_number") or "",
        "website": website,
        "website_trusted": website_trusted,
        "emails": emails,
        "socials": socials,
        "rating": lead.get("rating"),
        "reviews": lead.get("reviews_count") or lead.get("review_count"),
        "category": lead.get("category") or lead.get("categories", [None])[0] if lead.get("categories") else "",
        "place_id": lead.get("place_id") or lead.get("placeId") or "",
    }


@tool
async def scrape_leads(
    query: str,
    location: str,
    max_results: int = 10,
    config: RunnableConfig = None,
) -> str:
    """Search Google Maps for businesses matching a query in a
    location. Args: query (business type), location (city +
    country), max_results (1-50)."""
    configurable = config.get("configurable", {}) if config else {}
    tenant_id = configurable.get("tenant_id", "default_tenant")

    run_input = {
        "searchStringsArray": [query],
        "locationQuery": location,
        "maxCrawledPlacesPerSearch": max_results,
        "scrapeContacts": True,
        "website": "withWebsite",
    }

    client = _get_client()
    run = await client.actor(ACTOR_ID).call(run_input=run_input)
    items = await client.dataset(run["defaultDatasetId"]).list_items()
    raw_leads = items.items

    leads = [_trim_lead(l) for l in raw_leads]
    leads = [l for l in leads if l.get("name") and (l.get("website") or l.get("phone") or l.get("emails"))]

    saved = save_leads_to_files(
        tenant_id=tenant_id,
        query=query,
        location=location,
        leads=leads,
    )

    import json
    return json.dumps({
        "count": len(leads),
        "leads": leads,
        "saved": {
            "files": {
                "csv": saved.get("csv"),
                "xlsx": saved.get("xlsx"),
                "count": saved.get("count", 0),
            },
        },
    }, ensure_ascii=False)


if __name__ == "__main__":
    import asyncio

    async def _test():
        result = await scrape_leads.ainvoke({
            "query": "real estate agencies",
            "location": "Mildura, Australia",
            "max_results": 5,
        })
        print(result)

    asyncio.run(_test())