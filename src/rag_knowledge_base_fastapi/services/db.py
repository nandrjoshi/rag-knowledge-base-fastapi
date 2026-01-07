from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from rag_knowledge_base_fastapi.config.settings import settings


@dataclass(frozen=True)
class DbHealth:
    ok: bool
    server_version: str | None = None
    error: str | None = None


def create_db_engine() -> Engine:
    """
    Create a SQLAlchemy Engine for Postgres.

    Notes:
    - Uses DATABASE_URL from settings (supports .env).
    - pool_pre_ping helps avoid stale connections in long-running apps.
    """
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
    )


def check_db_health(engine: Engine) -> DbHealth:
    """
    Lightweight connectivity check.
    Does NOT require any tables—only validates we can connect and run a simple query.
    """
    try:
        with engine.connect() as conn:
            version = conn.execute(text("select version()")).scalar_one()
        return DbHealth(ok=True, server_version=str(version))
    except Exception as exc:  # intentionally broad for a health check
        return DbHealth(ok=False, error=str(exc))
