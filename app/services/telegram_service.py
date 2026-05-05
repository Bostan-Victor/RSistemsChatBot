from __future__ import annotations

import logging
import os

import requests

from ..models.lead import Lead

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"


def send_lead_notification(lead: Lead) -> None:
    """Send a Telegram message with lead details to the configured chat.

    Raises ValueError if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set.
    Raises requests.RequestException on network/API failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set.")

    text = (
        f"Lead nou RSistems:\n"
        f"Nume: {lead.name}\n"
        f"Telefon: {lead.phone or '—'}\n"
        f"Email: {lead.email or '—'}\n"
        f"Business: {lead.business_type or '—'}\n"
        f"Nume Business: {lead.business_name or '—'}\n"
        f"Nr. Locații: {lead.locations_count or '—'}"
    )

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    response.raise_for_status()
