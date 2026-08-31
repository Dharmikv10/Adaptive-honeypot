"""
db/database.py

Shared SQLite logging layer for the adaptive honeypot.

Phase 3 adds session tracking on top of the Phase 1/2 event log:
  - attack_logs   : every individual event (connection, auth_attempt, command,
                     probe, error), now tagged with a session_id
  - sessions      : one row per connection, with aggregate stats computed at
                     session close (duration, event counts, unique creds,
                     timing) -- this is what feature_extraction.py reads from
                     to build ML feature vectors, and what dashboard "Attack
                     Replay" (Phase 5) will reconstruct from attack_logs
                     filtered by session_id.
"""

import asyncio
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("/app/data/honeypot.db")

# This lock only synchronizes writes *within this one process*. Each service
# runs in its own container/process, so it does nothing across containers --
# that's what WAL mode + busy_timeout below are for.
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db():
    """Create all tables if they don't exist yet. Call once at startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attack_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,
            timestamp     TEXT NOT NULL,
            source_ip     TEXT NOT NULL,
            source_port   INTEGER,
            dest_port     INTEGER,
            protocol      TEXT NOT NULL,
            event_type    TEXT,          -- e.g. 'auth_attempt', 'connection', 'command'
            username      TEXT,
            password      TEXT,
            payload       TEXT,          -- raw command / request body / extra detail
            notes         TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id           TEXT PRIMARY KEY,
            source_ip            TEXT NOT NULL,
            protocol              TEXT NOT NULL,
            dest_port             INTEGER,
            start_time            TEXT NOT NULL,
            end_time              TEXT,
            duration_seconds      REAL,
            event_count           INTEGER DEFAULT 0,
            auth_attempts         INTEGER DEFAULT 0,
            unique_usernames      INTEGER DEFAULT 0,
            unique_passwords      INTEGER DEFAULT 0,
            command_count         INTEGER DEFAULT 0,
            avg_seconds_between   REAL,
            -- filled in later by the ML + adaptation layers (Phase 3 / 4):
predicted_class       TEXT,
            confidence            REAL,
            reason                TEXT,
            adaptation_taken      TEXT,
            -- bounded 0-10 threat score for THIS session specifically,
            -- snapshotted at decision time (see ml/adaption.py):
            threat_score          REAL,
            -- optional ground-truth label for training data (Phase 3 synthetic
            -- data sets this directly; real sessions get labeled manually or
            -- left NULL and used only for prediction, not training):
            label                 TEXT
        )
        """
    )
    # Migration for DBs created before threat_score existed on sessions.
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN threat_score REAL")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ip_policy (
            source_ip           TEXT PRIMARY KEY,
            threat_score         REAL DEFAULT 0,
            tarpit                INTEGER DEFAULT 0,
            escalated_logging     INTEGER DEFAULT 0,
            decoy_expanded        INTEGER DEFAULT 0,
            banner_override       TEXT,
            times_flagged         INTEGER DEFAULT 0,
            last_classification   TEXT,
            last_updated          TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON attack_logs(session_id)")
    conn.commit()
    conn.close()
    print(f"[db] Initialized SQLite database at {DB_PATH}")


def get_policy(source_ip: str) -> dict:
    """Read the current adaptive policy for an IP. Returns sane defaults
    (no tarpit, no escalation, no override) if the IP has never been flagged."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM ip_policy WHERE source_ip = ?", (source_ip,)).fetchone()
    conn.close()
    if row is None:
        return {
            "source_ip": source_ip, "threat_score": 0, "tarpit": 0,
            "escalated_logging": 0, "decoy_expanded": 0, "banner_override": None,
            "times_flagged": 0, "last_classification": None,
        }
    return dict(row)


def update_policy(source_ip: str, **fields):
    """
    Upsert fields onto an IP's policy row. Called by ml/adaptation.py after
    a classification decision. `fields` can include any column from
    ip_policy except source_ip -- e.g.:
        update_policy(ip, tarpit=1, threat_score=7.5, times_flagged=3)
    """
    with _lock:
        conn = _connect()
        existing = conn.execute(
            "SELECT source_ip FROM ip_policy WHERE source_ip = ?", (source_ip,)
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        fields["last_updated"] = now

        if existing is None:
            columns = ["source_ip"] + list(fields.keys())
            placeholders = ", ".join("?" for _ in columns)
            values = [source_ip] + list(fields.values())
            conn.execute(
                f"INSERT INTO ip_policy ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        else:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE ip_policy SET {set_clause} WHERE source_ip = ?",
                list(fields.values()) + [source_ip],
            )
        conn.commit()
        conn.close()


async def get_policy_async(source_ip: str) -> dict:
    return await asyncio.to_thread(get_policy, source_ip)


def record_session_intelligence(session_id: str, predicted_class: str = None,
                                 confidence: float = None, reason: str = None,
                                 adaptation_taken: str = None, threat_score: float = None):
    """Write ML classification + adaptation decision back onto a session row.
    threat_score is THIS session's own bounded (0-10) score, snapshotted at
    decision time -- do not re-read it later from ip_policy, which changes
    as the IP sends more traffic."""
    with _lock:
        conn = _connect()
        conn.execute(
            """UPDATE sessions SET predicted_class = ?, confidence = ?,
                 reason = ?, adaptation_taken = ?, threat_score = ? WHERE session_id = ?""",
            (predicted_class, confidence, reason, adaptation_taken, threat_score, session_id),
        )
        conn.commit()
        conn.close()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def start_session(source_ip: str, protocol: str, dest_port: int = None) -> str:
    """Call once when a connection opens. Returns the session_id to pass to
    every log_attempt() call for that connection, and to end_session() later."""
    session_id = new_session_id()
    start_time = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        conn.execute(
            """INSERT INTO sessions (session_id, source_ip, protocol, dest_port, start_time)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, source_ip, protocol, dest_port, start_time),
        )
        conn.commit()
        conn.close()
    return session_id


def end_session(session_id: str):
    """
    Call once when a connection closes. Computes aggregate stats from every
    attack_logs row tagged with this session_id and writes them into the
    sessions table -- this is the row feature_extraction.py will read.
    """
    with _lock:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM attack_logs WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

        if not rows:
            conn.close()
            return

        start_row = conn.execute(
            "SELECT start_time FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        start_time = datetime.fromisoformat(start_row[0]) if start_row else \
            datetime.fromisoformat(rows[0]["timestamp"])
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

        usernames = {r["username"] for r in rows if r["username"]}
        passwords = {r["password"] for r in rows if r["password"]}
        auth_attempts = sum(1 for r in rows if r["event_type"] == "auth_attempt")
        command_count = sum(1 for r in rows if r["event_type"] in ("command", "probe"))

        conn.execute(
            """UPDATE sessions SET
                 end_time = ?, duration_seconds = ?, event_count = ?,
                 auth_attempts = ?, unique_usernames = ?, unique_passwords = ?,
                 command_count = ?, avg_seconds_between = ?
               WHERE session_id = ?""",
            (
                end_time.isoformat(), duration, len(rows),
                auth_attempts, len(usernames), len(passwords),
                command_count, avg_gap, session_id,
            ),
        )
        conn.commit()
        conn.close()


def log_attempt(
    source_ip: str,
    protocol: str,
    dest_port: int = None,
    source_port: int = None,
    event_type: str = "connection",
    username: str = None,
    password: str = None,
    payload: str = None,
    notes: str = None,
    session_id: str = None,
):
    """Insert one attack event. Safe to call from multiple threads/services."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO attack_logs
                (session_id, timestamp, source_ip, source_port, dest_port, protocol,
                 event_type, username, password, payload, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, timestamp, source_ip, source_port, dest_port, protocol,
             event_type, username, password, payload, notes),
        )
        conn.commit()
        conn.close()
    print(f"[db] {protocol} | {source_ip} | {event_type} | user={username} pass={password}")


async def log_attempt_async(*args, **kwargs):
    """
    Non-blocking version of log_attempt for use inside asyncio services
    (fake_ftp, fake_telnet, fake_db, and the async fake_http/fake_ssh wrappers).
    Runs the actual (blocking) SQLite write in a worker thread so it never
    stalls the event loop while other decoy connections are being served.
    """
    await asyncio.to_thread(log_attempt, *args, **kwargs)


async def end_session_async(session_id: str):
    await asyncio.to_thread(end_session, session_id)


if __name__ == "__main__":
    # Quick manual test: python3 db/database.py
    init_db()
    sid = start_session("127.0.0.1", "TEST", dest_port=2222)
    log_attempt("127.0.0.1", "TEST", dest_port=2222, event_type="self_test",
                username="test", password="test123", session_id=sid)
    end_session(sid)
    print("Test session recorded:", sid)