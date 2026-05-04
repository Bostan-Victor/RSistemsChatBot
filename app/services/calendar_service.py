from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build

_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_SLOT_DURATION_HOURS = 1
_WORK_START = time(10, 0)
_WORK_END = time(18, 0)
_WORK_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon–Fri


def _build_service():
    json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "")
    if not json_path:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON_PATH is not set.")
    if not os.path.isabs(json_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(project_root, json_path)
    if not os.path.isfile(json_path):
        raise ValueError(
            f"Service account file not found at: {json_path}"
        )
    creds = service_account.Credentials.from_service_account_file(
        json_path, scopes=_SCOPES
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _all_slots_for_date(d: date) -> list[datetime]:
    slots: list[datetime] = []
    current = datetime.combine(d, _WORK_START)
    end_boundary = datetime.combine(d, _WORK_END)
    while current + timedelta(hours=_SLOT_DURATION_HOURS) <= end_boundary:
        slots.append(current)
        current += timedelta(hours=_SLOT_DURATION_HOURS)
    return slots


def get_available_slots(query_date: date) -> list[str]:
    """Return ISO-format datetime strings for free 1-hour slots on query_date.

    Slots are in the local naive datetime (no tzinfo) — the frontend displays them as-is.
    Returns an empty list for weekends.
    """
    if query_date.weekday() not in _WORK_WEEKDAYS:
        return []

    all_slots = _all_slots_for_date(query_date)
    if not all_slots:
        return []

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = _build_service()

    day_start = datetime.combine(query_date, time(0, 0)).replace(tzinfo=timezone.utc)
    day_end = datetime.combine(query_date, time(23, 59, 59)).replace(tzinfo=timezone.utc)

    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    busy: list[tuple[datetime, datetime]] = []
    for ev in events:
        start_str = ev.get("start", {}).get("dateTime")
        end_str = ev.get("end", {}).get("dateTime")
        if not start_str or not end_str:
            continue
        ev_start = datetime.fromisoformat(start_str).replace(tzinfo=None)
        ev_end = datetime.fromisoformat(end_str).replace(tzinfo=None)
        busy.append((ev_start, ev_end))

    now = datetime.now()
    free: list[str] = []
    for slot in all_slots:
        slot_end = slot + timedelta(hours=_SLOT_DURATION_HOURS)
        if slot <= now:
            continue
        overlap = any(
            not (slot_end <= b_start or slot >= b_end) for b_start, b_end in busy
        )
        if not overlap:
            free.append(slot.isoformat())

    return free


def create_event(
    *,
    name: str,
    email: str,
    phone: str,
    business_type: str,
    start_dt: datetime,
) -> str:
    """Create a 1-hour Google Calendar event and return its event ID."""
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = _build_service()

    end_dt = start_dt + timedelta(hours=_SLOT_DURATION_HOURS)

    event_body = {
        "summary": f"Demo Showroom — {name} ({business_type})",
        "description": (
            f"Rezervare demo showroom RSistems\n\n"
            f"Nume: {name}\n"
            f"Telefon: {phone}\n"
            f"Email: {email}\n"
            f"Tip afacere: {business_type}"
        ),
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "Europe/Bucharest"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "Europe/Bucharest"},
        "attendees": [{"email": email}],
    }

    created = service.events().insert(calendarId=calendar_id, body=event_body).execute()
    return created.get("id", "")
