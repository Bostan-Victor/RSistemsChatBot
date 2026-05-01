from __future__ import annotations

from app import create_app
from app.extensions import db
from sqlalchemy import inspect, text


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()

        # Lightweight schema patching for local dev (no migrations folder yet).
        # If the DB already exists, ensure new columns are added.
        try:
            inspector = inspect(db.engine)
            if "leads" in inspector.get_table_names():
                columns = {c["name"] for c in inspector.get_columns("leads")}
                if "business_name" not in columns:
                    db.session.execute(text("ALTER TABLE leads ADD COLUMN business_name VARCHAR(200)"))
                    db.session.commit()
                if "lead_score" not in columns:
                    db.session.execute(text("ALTER TABLE leads ADD COLUMN lead_score INTEGER"))
                    db.session.commit()
        except Exception:
            # If anything goes wrong here (non-SQLite DB, permissions, etc.), keep init_db simple.
            db.session.rollback()
    print("DB initialized (tables created if missing).")


if __name__ == "__main__":
    main()
