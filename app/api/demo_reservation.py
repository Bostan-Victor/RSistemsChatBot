from __future__ import annotations

import logging
from datetime import datetime

from flask import Blueprint, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..models.demo_reservation import DemoReservation
from ..services.calendar_service import create_event, get_available_slots
from ..services.email_service import send_demo_confirmation
from ..services.validators import is_valid_email, is_valid_phone

bp = Blueprint("demo_reservation", __name__)
log = logging.getLogger(__name__)

_BUSINESS_TYPES = [
    "Restaurant",
    "Cafenea",
    "Bar / Pub",
    "Fast-food",
    "Delivery / Takeaway",
    "Lanț de locații",
]


@bp.get("/demo-reservation/slots")
def get_slots():
    date_str = request.args.get("date", "").strip()
    if not date_str:
        return jsonify({"error": "invalid_request", "message": "'date' query param required (YYYY-MM-DD)."}), 400

    try:
        query_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "invalid_request", "message": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        slots = get_available_slots(query_date)
    except ValueError as exc:
        return jsonify({"error": "config_error", "message": str(exc)}), 500
    except Exception as exc:
        log.exception("Error fetching calendar slots")
        return jsonify({"error": "calendar_error", "message": str(exc)}), 500

    return jsonify({"date": date_str, "slots": slots})


@bp.post("/demo-reservation")
def demo_reservation():
    payload = request.get_json(silent=True) or {}

    name = (payload.get("name") or "").strip()
    phone = (payload.get("phone") or "").strip()
    email = (payload.get("email") or "").strip()
    business_type = (payload.get("business_type") or "").strip()
    datetime_str = (payload.get("datetime") or "").strip()

    if not name or len(name) < 2:
        return jsonify({"error": "invalid_request", "message": "'name' is required (min 2 chars)."}), 400
    if not phone or not is_valid_phone(phone):
        return jsonify({"error": "invalid_request", "message": "'phone' is invalid."}), 400
    if not email or not is_valid_email(email):
        return jsonify({"error": "invalid_request", "message": "'email' is invalid."}), 400
    if business_type not in _BUSINESS_TYPES:
        return jsonify({"error": "invalid_request", "message": f"'business_type' must be one of: {_BUSINESS_TYPES}."}), 400
    if not datetime_str:
        return jsonify({"error": "invalid_request", "message": "'datetime' is required (ISO format)."}), 400

    try:
        reserved_dt = datetime.fromisoformat(datetime_str)
    except ValueError:
        return jsonify({"error": "invalid_request", "message": "Invalid 'datetime' format. Use ISO 8601."}), 400

    if reserved_dt <= datetime.now():
        return jsonify({"error": "invalid_request", "message": "Selected datetime is in the past."}), 400

    try:
        google_event_id = create_event(
            name=name,
            email=email,
            phone=phone,
            business_type=business_type,
            start_dt=reserved_dt,
        )
    except ValueError as exc:
        return jsonify({"error": "config_error", "message": str(exc)}), 500
    except Exception as exc:
        log.exception("Error creating calendar event")
        return jsonify({"error": "calendar_error", "message": str(exc)}), 500

    try:
        reservation = DemoReservation(
            name=name,
            phone=phone,
            email=email,
            business_type=business_type,
            reserved_datetime=reserved_dt,
            google_event_id=google_event_id,
        )
        db.session.add(reservation)
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        log.exception("Error saving demo reservation to DB")
        return jsonify({"error": "db_error", "message": str(exc)}), 500

    try:
        send_demo_confirmation(
            to_email=email,
            name=name,
            business_type=business_type,
            reserved_datetime=reserved_dt,
        )
    except Exception as exc:
        log.warning("Confirmation email failed (reservation still saved): %s", exc)

    return jsonify({
        "status": "ok",
        "message": "Rezervarea a fost confirmată! Veți primi un email de confirmare.",
        "reservation_id": reservation.id,
    })

