import logging
import os
from datetime import datetime, timezone
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

EXPORTS_DIR = "exports"

COLUMNS = [
    "name", "address", "phone", "website", "website_trusted", "emails",
    "facebook", "instagram", "linkedin", "twitter", "tiktok", "youtube",
    "rating", "reviews", "category", "place_id",
    "query", "location", "scraped_at",
]


def _flatten_row(lead: dict, query: str, location: str, scraped_at: str) -> dict:
    socials = lead.get("socials") or {}
    emails = lead.get("emails") or []
    return {
        "name": lead.get("name") or "",
        "address": lead.get("address") or "",
        "phone": lead.get("phone") or "",
        "website": lead.get("website") or "",
        "website_trusted": lead.get("website_trusted", False),
        "emails": "; ".join(emails) if isinstance(emails, list) else (emails or ""),
        "facebook": socials.get("facebook") or "",
        "instagram": socials.get("instagram") or "",
        "linkedin": socials.get("linkedin") or "",
        "twitter": socials.get("twitter") or "",
        "tiktok": socials.get("tiktok") or "",
        "youtube": socials.get("youtube") or "",
        "rating": lead.get("rating") if lead.get("rating") is not None else "",
        "reviews": lead.get("reviews") if lead.get("reviews") is not None else "",
        "category": lead.get("category") or "",
        "place_id": lead.get("place_id") or "",
        "query": query,
        "location": location,
        "scraped_at": scraped_at,
    }


def save_leads_to_files(
    tenant_id: str,
    query: str,
    location: str,
    leads: list[dict],
) -> dict:
    """
    Save leads to per-tenant CSV + XLSX files.
    Writes timestamped files plus latest.{csv,xlsx}.
    Returns: {"csv": path, "xlsx": path, "count": N}
    """
    if not leads:
        return {"csv": None, "xlsx": None, "count": 0}

    tenant_dir = os.path.join(EXPORTS_DIR, tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scraped_at = datetime.now(timezone.utc).isoformat()

    rows = [_flatten_row(l, query, location, scraped_at) for l in leads]
    df = pd.DataFrame(rows, columns=COLUMNS)

    csv_path = os.path.join(tenant_dir, f"leads_{ts}.csv")
    xlsx_path = os.path.join(tenant_dir, f"leads_{ts}.xlsx")
    latest_csv = os.path.join(tenant_dir, "latest.csv")
    latest_xlsx = os.path.join(tenant_dir, "latest.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    df.to_excel(latest_xlsx, index=False)

    logger.info(
        "save_leads_to_files: tenant=%s count=%d csv=%s xlsx=%s",
        tenant_id, len(rows), csv_path, xlsx_path,
    )

    return {"csv": csv_path, "xlsx": xlsx_path, "count": len(rows)}