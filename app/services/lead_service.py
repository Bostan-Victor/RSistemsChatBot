from __future__ import annotations

from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..models.lead import Lead
from .validators import is_valid_email, is_valid_phone, parse_locations_count_strict

class LeadService:
    """Lead capture service used by both API and chat flow."""

    @staticmethod
    def _score_for_locations(locations_count: int) -> int:
        if locations_count <= 1:
            return 1
        if 2 <= locations_count <= 5:
            return 2
        return 3

    @staticmethod
    def create_lead(payload: dict[str, Any]) -> Lead:
        name = (payload.get("name") or "").strip()
        business_name = (payload.get("business_name") or "").strip() or None
        phone = (payload.get("phone") or "").strip() or None
        email = (payload.get("email") or "").strip() or None
        business_type = (payload.get("type_of_business") or payload.get("business_type") or "").strip() or None
        locations_count_raw = payload.get("nr_of_locations")
        if locations_count_raw is None:
            locations_count_raw = payload.get("locations_count")

        if not name:
            raise ValueError("'name' is required")

        if not phone:
            raise ValueError("'phone' is required")

        if not email:
            raise ValueError("'email' is required")

        if not business_name:
            raise ValueError("'business_name' is required")

        if not is_valid_phone(phone):
            raise ValueError("'phone' is invalid")

        if not is_valid_email(email):
            raise ValueError("'email' is invalid")

        locations_count: int | None = None
        if locations_count_raw in (None, ""):
            raise ValueError("'nr_of_locations' is required")

        locations_count = parse_locations_count_strict(locations_count_raw)
        if locations_count is None:
            raise ValueError("'nr_of_locations' must be an integer")
        if locations_count <= 0:
            raise ValueError("'nr_of_locations' must be >= 1")

        lead_score = LeadService._score_for_locations(locations_count)

        lead = Lead(
            name=name,
            business_name=business_name,
            phone=phone,
            email=email,
            business_type=business_type,
            locations_count=locations_count,
            lead_score=lead_score,
        )

        try:
            db.session.add(lead)
            db.session.commit()
        except SQLAlchemyError as exc:
            db.session.rollback()
            raise exc

        return lead
