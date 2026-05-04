from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from .chat import bp as chat_bp
    from .health import bp as health_bp
    from .leads import bp as leads_bp
    from .demo_reservation import bp as demo_reservation_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(leads_bp, url_prefix="/api")
    app.register_blueprint(demo_reservation_bp, url_prefix="/api")
