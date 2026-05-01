from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..models.lead import Lead
from ..services.lead_service import LeadService


bp = Blueprint("leads", __name__)


@bp.get("/leads")
def list_leads():
	leads = Lead.query.order_by(Lead.created_at.desc()).all()
	return jsonify(
		[
			{
				"id": lead.id,
				"name": lead.name,
				"business_name": lead.business_name,
				"phone": lead.phone,
				"email": lead.email,
				"type_of_business": lead.business_type,
				"nr_of_locations": lead.locations_count,
				"lead_score": lead.lead_score,
				"created_at": lead.created_at.isoformat() + "Z",
			}
			for lead in leads
		]
	)


@bp.post("/leads")

def create_lead():
	payload = request.get_json(silent=True) or {}

	try:
		lead = LeadService.create_lead(payload)
	except ValueError as exc:
		return (
			jsonify({"error": "invalid_request", "message": str(exc)}),
			400,
		)

	return (
		jsonify(
			{
				"id": lead.id,
				"created_at": lead.created_at.isoformat() + "Z",
			}
		),
		201,
	)

