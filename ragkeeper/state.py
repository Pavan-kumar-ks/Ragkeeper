import sqlite3
from pathlib import Path

FILE_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
    source_path TEXT PRIMARY KEY,
    file_content_hash TEXT NOT NULL,
    commit_hash TEXT,
    chunk_count INTEGER NOT NULL,
    last_synced_at TEXT NOT NULL
)
"""

SYNC_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    commit_hash TEXT,
    files_added INTEGER,
    files_updated INTEGER,
    files_deleted INTEGER,
    files_unchanged INTEGER,
    chunks_added INTEGER,
    chunks_deleted INTEGER,
    duration_s REAL,
    status TEXT NOT NULL,
    error TEXT
)
"""


def init_db(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(FILE_STATE_SCHEMA)
    conn.execute(SYNC_RUNS_SCHEMA)
    conn.commit()
    return conn


def reset(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS file_state")
    conn.execute(FILE_STATE_SCHEMA)
    conn.commit()


def load_file_state(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT source_path, file_content_hash, commit_hash, chunk_count, last_synced_at FROM file_state"
    ).fetchall()
    return {
        row[0]: {
            "file_content_hash": row[1],
            "commit_hash": row[2],
            "chunk_count": row[3],
            "last_synced_at": row[4],
        }
        for row in rows
    }


def upsert_file_state(
    conn: sqlite3.Connection,
    source_path: str,
    file_content_hash: str,
    commit_hash: str,
    chunk_count: int,
    synced_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO file_state (source_path, file_content_hash, commit_hash, chunk_count, last_synced_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            file_content_hash = excluded.file_content_hash,
            commit_hash = excluded.commit_hash,
            chunk_count = excluded.chunk_count,
            last_synced_at = excluded.last_synced_at
        """,
        (source_path, file_content_hash, commit_hash, chunk_count, synced_at),
    )
    conn.commit()


def delete_file_state(conn: sqlite3.Connection, source_path: str) -> None:
    conn.execute("DELETE FROM file_state WHERE source_path = ?", (source_path,))
    conn.commit()


def record_sync_run(
    conn: sqlite3.Connection,
    started_at: str,
    finished_at: str,
    commit_hash: str | None,
    files_added: int,
    files_updated: int,
    files_deleted: int,
    files_unchanged: int,
    chunks_added: int,
    chunks_deleted: int,
    duration_s: float,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sync_runs (
            started_at, finished_at, commit_hash, files_added, files_updated, files_deleted,
            files_unchanged, chunks_added, chunks_deleted, duration_s, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started_at,
            finished_at,
            commit_hash,
            files_added,
            files_updated,
            files_deleted,
            files_unchanged,
            chunks_added,
            chunks_deleted,
            duration_s,
            status,
            error,
        ),
    )
    conn.commit()


def get_latest_sync_run(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        """
        SELECT started_at, finished_at, commit_hash, files_added, files_updated, files_deleted,
               files_unchanged, chunks_added, chunks_deleted, duration_s, status, error
        FROM sync_runs ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    keys = [
        "started_at", "finished_at", "commit_hash", "files_added", "files_updated", "files_deleted",
        "files_unchanged", "chunks_added", "chunks_deleted", "duration_s", "status", "error",
    ]
    return dict(zip(keys, row))


def get_sync_run_history(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT started_at, finished_at, commit_hash, files_added, files_updated, files_deleted,
               files_unchanged, chunks_added, chunks_deleted, duration_s, status, error
        FROM sync_runs ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    keys = [
        "started_at", "finished_at", "commit_hash", "files_added", "files_updated", "files_deleted",
        "files_unchanged", "chunks_added", "chunks_deleted", "duration_s", "status", "error",
    ]
    return [dict(zip(keys, row)) for row in rows]
