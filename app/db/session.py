"""Async engine / session factory, dev-time schema creation and SQL migration runner."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from app.config.settings import Settings, get_settings
from app.core.errors import QuantCoreError
from app.db.models import Base, json_default

logger = logging.getLogger(__name__)

DEFAULT_MIGRATIONS_PATH = Path(__file__).resolve().parents[2] / "migrations"

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    checksum   TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class MigrationError(QuantCoreError):
    """A SQL migration could not be applied or an applied one was modified."""


def _json_serializer(obj: Any) -> str:
    # ensure_ascii=False keeps Turkish text readable in JSONB (e.g. risk reasons).
    return json.dumps(obj, default=json_default, ensure_ascii=False)


def create_engine_from_settings(settings: Settings | None = None, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine from ``Settings.database_url``.

    PostgreSQL gets a pre-pinged pool; SQLite (tests / dev) gets foreign keys enabled and,
    for in-memory databases, a single shared connection so every session sees the same data.
    """
    settings = settings or get_settings()
    return create_engine_from_url(settings.database_url, echo=echo)


def create_engine_from_url(database_url: str, *, echo: bool = False) -> AsyncEngine:
    url = make_url(database_url)
    kwargs: dict[str, Any] = {"echo": echo, "json_serializer": _json_serializer}

    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:"):
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_async_engine(url, **kwargs)

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    kwargs.update(pool_pre_ping=True, pool_size=10, max_overflow=10, pool_recycle=1800)
    return create_async_engine(url, **kwargs)


def async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory; ``expire_on_commit=False`` so ORM rows stay readable after commit."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def get_session(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, rollback on any exception."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine) -> None:
    """``Base.metadata.create_all`` for dev / SQLite.

    On PostgreSQL production use :func:`apply_sql_migrations` instead: ``create_all`` does
    not create hypertables, extensions, triggers or policies.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _discover_migrations(path: Path) -> list[Path]:
    if not path.is_dir():
        raise MigrationError(f"Migrations directory not found: {path}")
    return sorted(p for p in path.glob("*.sql") if p.is_file())


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


async def apply_sql_migrations(engine: AsyncEngine, path: Path | str | None = None) -> list[str]:
    """Run ``<path>/*.sql`` in filename order on PostgreSQL; return versions applied now.

    * Each file runs in its own transaction together with its ``schema_migrations`` row.
    * An already applied file whose content changed raises :class:`MigrationError`
      (write a new numbered migration instead of editing an applied one).
    * Non-PostgreSQL engines are skipped (SQLite uses :func:`init_db`).
    """
    if engine.dialect.name != "postgresql":
        logger.info("apply_sql_migrations skipped: dialect %s is not PostgreSQL", engine.dialect.name)
        return []

    files = _discover_migrations(Path(path) if path else DEFAULT_MIGRATIONS_PATH)

    async with engine.begin() as conn:
        await conn.execute(text(_SCHEMA_MIGRATIONS_DDL))
        rows = (await conn.execute(text("SELECT version, checksum FROM schema_migrations"))).all()
    applied = {version: checksum for version, checksum in rows}

    newly_applied: list[str] = []
    for file in files:
        version = file.stem
        sql = file.read_text(encoding="utf-8")
        checksum = _checksum(sql)
        if version in applied:
            if applied[version] != checksum:
                raise MigrationError(f"Migration {version} was modified after being applied; add a new migration instead.")
            continue

        logger.info("Applying migration %s", version)
        try:
            async with engine.connect() as conn:
                raw = await conn.get_raw_connection()
                driver = raw.driver_connection  # asyncpg.Connection
                # asyncpg's execute() without arguments uses the simple-query protocol, which
                # allows many statements (DDL, DO blocks, $$ functions) in one call.
                async with driver.transaction():
                    await driver.execute(sql)
                    await driver.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)", version, checksum
                    )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise MigrationError(f"Migration {version} failed: {exc}") from exc
        newly_applied.append(version)

    return newly_applied
