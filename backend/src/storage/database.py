from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ---------------------------------------------------------------------------
# Shared metadata – single source of truth for all table definitions
# ---------------------------------------------------------------------------

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("file_name", String, nullable=False),
    Column("case_name", String, nullable=True),
    Column("court_name", String, nullable=True),
    Column("judgment_date", String, nullable=True),
    Column("page_count", Integer, nullable=False, server_default="0"),
    Column("char_count", Integer, nullable=False, server_default="0"),
    Column("ingested_at", String, nullable=False),
    Column("status", String, nullable=False, server_default="'success'"),
    UniqueConstraint("file_name", name="uq_documents_file_name"),
)

Index("idx_documents_status", documents_table.c.status)
Index("idx_documents_file_name", documents_table.c.file_name)

chunks_table = Table(
    "chunks",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "document_id",
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("content", Text, nullable=False),
    Column("char_start", Integer, nullable=False),
    Column("char_end", Integer, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("embedded_at", String, nullable=False),
    Column("section", String, nullable=False, server_default="'other'"),
    Column("chunk_type", String, nullable=False, server_default="'child'"),
    Column("parent_id", String, nullable=True),
    UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_position"),
)

Index("idx_chunks_document_id", chunks_table.c.document_id)
Index("idx_chunks_parent_id", chunks_table.c.parent_id)
Index("idx_chunks_type", chunks_table.c.chunk_type)

schema_migrations_table = Table(
    "_schema_migrations",
    metadata,
    Column("migration_id", String, primary_key=True),
)

ingestion_runs_table = Table(
    "ingestion_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("corpus_dir", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("completed_at", String, nullable=True),
    Column("total_files", Integer, nullable=False, server_default="0"),
    Column("succeeded", Integer, nullable=False, server_default="0"),
    Column("failed", Integer, nullable=False, server_default="0"),
    Column("total_chunks", Integer, nullable=False, server_default="0"),
    Column("status", String, nullable=False, server_default="'running'"),
)

ingestion_failures_table = Table(
    "ingestion_failures",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "run_id",
        String,
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("file_name", String, nullable=False),
    Column("error_message", String, nullable=False),
)

Index("idx_failures_run_id", ingestion_failures_table.c.run_id)

messages_table = Table(
    "messages",
    metadata,
    Column("id", String, primary_key=True),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("raw_response", Text, nullable=True),
    Column("agent_steps", Text, nullable=True),
    Column("query_type", String, nullable=True),
    Column("sources_searched", Integer, nullable=False, server_default="0"),
    Column("created_at", String, nullable=False),
    CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
)

Index("idx_messages_created_at", messages_table.c.created_at)

# ---------------------------------------------------------------------------
# FTS5 virtual table + triggers (raw DDL – SQLAlchemy has no FTS5 dialect)
# ---------------------------------------------------------------------------

_FTS5_DDL = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        content='chunks',
        content_rowid='rowid'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
]

# ---------------------------------------------------------------------------
# Column-level migrations (idempotent – tracked in _schema_migrations)
# ---------------------------------------------------------------------------

_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    (
        "add_chunks_section",
        "ALTER TABLE chunks ADD COLUMN section TEXT NOT NULL DEFAULT 'other'",
    ),
    (
        "add_chunks_chunk_type",
        "ALTER TABLE chunks ADD COLUMN chunk_type TEXT NOT NULL DEFAULT 'child'",
    ),
    (
        "add_chunks_parent_id",
        "ALTER TABLE chunks ADD COLUMN parent_id TEXT",
    ),
    (
        "add_messages_raw_response",
        "ALTER TABLE messages ADD COLUMN raw_response TEXT",
    ),
    (
        "add_messages_agent_steps",
        "ALTER TABLE messages ADD COLUMN agent_steps TEXT",
    ),
]

# Multi-statement migrations that rebuild tables.
# Each entry is (migration_id, [sql_statement, ...]).
_TABLE_REBUILD_MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "flatten_to_single_user_chat",
        [
            """
            CREATE TABLE IF NOT EXISTS messages_new (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                raw_response TEXT,
                query_type TEXT,
                sources_searched INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """,
            """
            INSERT OR IGNORE INTO messages_new
                (id, role, content, raw_response, query_type, sources_searched, created_at)
            SELECT id, role, content, raw_response, query_type, sources_searched, created_at
            FROM messages
            """,
            "DROP TABLE messages",
            "ALTER TABLE messages_new RENAME TO messages",
            "DROP TABLE IF EXISTS conversations",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
        ],
    ),
]

_INDEX_MIGRATIONS: list[tuple[str, str]] = [
    (
        "idx_chunks_parent_id",
        "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)",
    ),
    (
        "idx_chunks_type",
        "CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)",
    ),
]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def create_db_engine(database_path: Path) -> AsyncEngine:
    """Create an async SQLAlchemy engine backed by SQLite + aiosqlite."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{database_path}"
    engine = create_async_engine(url, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _connection_record) -> None:  # type: ignore[type-arg]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a reusable factory that yields :class:`AsyncSession` instances."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


async def init_schema(engine: AsyncEngine) -> None:
    """Create all tables (idempotent), apply FTS5 DDL and column migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        for ddl in _FTS5_DDL:
            await conn.execute(text(ddl))

    await _run_migrations(engine)


async def _run_migrations(engine: AsyncEngine) -> None:
    """Apply idempotent ALTER TABLE / CREATE INDEX migrations."""
    for migration_id, statements in _TABLE_REBUILD_MIGRATIONS:
        async with engine.begin() as conn:
            row = await conn.execute(
                text("SELECT 1 FROM _schema_migrations WHERE migration_id = :mid"),
                {"mid": migration_id},
            )
            if row.first() is None:
                for sql in statements:
                    await conn.execute(text(sql))
                await conn.execute(
                    text("INSERT INTO _schema_migrations (migration_id) VALUES (:mid)"),
                    {"mid": migration_id},
                )

    for migration_id, sql in _COLUMN_MIGRATIONS:
        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT 1 FROM _schema_migrations WHERE migration_id = :mid"),
                {"mid": migration_id},
            )
            if row.first() is None:
                try:
                    await conn.execute(text(sql))
                    await conn.execute(
                        text("INSERT INTO _schema_migrations (migration_id) VALUES (:mid)"),
                        {"mid": migration_id},
                    )
                    await conn.commit()
                except Exception:
                    await conn.rollback()

    for migration_id, sql in _INDEX_MIGRATIONS:
        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT 1 FROM _schema_migrations WHERE migration_id = :mid"),
                {"mid": migration_id},
            )
            if row.first() is None:
                await conn.execute(text(sql))
                await conn.execute(
                    text("INSERT INTO _schema_migrations (migration_id) VALUES (:mid)"),
                    {"mid": migration_id},
                )
                await conn.commit()
