from __future__ import annotations

from datetime import datetime

from ..extensions import db


class DemoReservation(db.Model):
    __tablename__ = "demo_reservations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    business_type = db.Column(db.String(100), nullable=False)
    reserved_datetime = db.Column(db.DateTime, nullable=False)
    google_event_id = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
