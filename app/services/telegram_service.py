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

    existing = "Da" if lead.has_existing_system is True else ("Nu" if lead.has_existing_system is False else "—")
    text = (
        f"Lead nou RSistems:\n"
        f"Nume: {lead.name}\n"
        f"Telefon: {lead.phone or '—'}\n"
        f"Email: {lead.email or '—'}\n"
        f"Business: {lead.business_type or '—'}\n"
        f"Nume Business: {lead.business_name or '—'}\n"
        f"Nr. Locații: {lead.locations_count or '—'}\n"
        f"Nr. Mese/POS: {lead.tables_count or '—'}\n"
        f"Sistem existent: {existing}\n"
        f"Oraș: {lead.city or '—'}\n"
        f"Preferință contact: {lead.contact_preference or '—'}"
    )

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    response.raise_for_status()


def send_support_notification(*, company: str, contact_name: str, phone: str, issue: str) -> None:
    """Send a Telegram message for a support ticket.

    Raises ValueError if env vars are missing.
    Raises requests.RequestException on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set.")

    text = (
        f"🔧 Solicitare SUPORT RSistems:\n"
        f"Companie: {company or '—'}\n"
        f"Persoană contact: {contact_name or '—'}\n"
        f"Telefon: {phone or '—'}\n"
        f"Problemă: {issue or '—'}"
    )

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    response.raise_for_status()


def send_transcript_document(*, lead_ref: str, messages: list[dict]) -> None:
    """Send the full conversation transcript as a .txt file to Telegram.

    Skips system messages. Labels turns as 'Client' / 'RSistems'.
    Raises ValueError if env vars are missing.
    Raises requests.RequestException on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set.")

    lines = [f"Transcript conversație — {lead_ref}", "=" * 44]
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        label = "Client" if role == "user" else "RSistems"
        lines.append(f"\n{label}:\n{m.get('content', '').strip()}")
    content = "\n".join(lines).encode("utf-8")

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendDocument",
        data={"chat_id": chat_id, "caption": f"📋 Transcript — {lead_ref}"},
        files={"document": (f"transcript_{lead_ref}.txt", content, "text/plain")},
        timeout=15,
    )
    response.raise_for_status()


def send_reservation_notification(
    *, name: str, phone: str, email: str, business_type: str, reserved_datetime: str
) -> None:
    """Send a Telegram message when a demo showroom reservation is created.

    Raises ValueError if env vars are missing.
    Raises requests.RequestException on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set.")

    text = (
        f"📅 Rezervare Demo Showroom RSistems:\n"
        f"Nume: {name or '—'}\n"
        f"Telefon: {phone or '—'}\n"
        f"Email: {email or '—'}\n"
        f"Tip afacere: {business_type or '—'}\n"
        f"Data și ora: {reserved_datetime or '—'}"
    )

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    response.raise_for_status()


def send_human_transfer_notification(*, name: str, phone: str, topic: str) -> None:
    """Send a Telegram message for a human manager transfer request.

    Raises ValueError if env vars are missing.
    Raises requests.RequestException on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set.")

    text = (
        f"👤 Transfer MANAGER RSistems:\n"
        f"Nume: {name or '—'}\n"
        f"Telefon: {phone or '—'}\n"
        f"Subiect: {topic or '—'}"
    )

    response = requests.post(
        f"{_API_BASE}/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    response.raise_for_status()
