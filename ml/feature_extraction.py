"""
ml/feature_extraction.py

Turns one honeypot session (identified by session_id) into a fixed-length
numeric feature vector the classifier can consume.

Design note: features are computed by aggregating attack_logs rows directly,
not by reading the (possibly not-yet-finalized) sessions table row. This
matters because HTTP sessions only get finalized by a background timeout,
so if you called this right after an attack -- exactly when a live demo
needs it -- the sessions table might still show zeros. Recomputing from
the raw log rows means feature extraction works correctly at any moment,
finalized or not.
"""

import math
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import DB_PATH  # single source of truth -- see db/database.py

# Fixed column order -- training and prediction MUST use this exact order.
FEATURE_COLUMNS = [
    "event_count",
    "auth_attempts",
    "unique_usernames",
    "unique_passwords",
    "command_count",
    "duration_seconds",
    "avg_seconds_between",
    "recent_session_count_from_ip",
    "username_entropy",
    "password_entropy",
    "repeated_password_ratio",
    "unique_payloads",
    "avg_payload_length",
    "commands_per_minute",
    "is_ssh",
    "is_http",
    "is_ftp",
    "is_telnet",
    "is_mysql",
]

PROTOCOLS = ["SSH", "HTTP", "FTP", "TELNET", "MYSQL"]
RECENT_WINDOW_SECONDS = 600  # 10 minutes, for the "connection frequency" feature


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _shannon_entropy(s: str) -> float:
    """Bits of entropy per character. 'admin' -> low, 'x7$K!q2' -> high.
    Used to tell a small fixed credential list (Brute Force, low entropy)
    apart from large randomized/leaked credential lists (Credential
    Stuffing, higher entropy)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _avg_entropy(values: set) -> float:
    if not values:
        return 0.0
    return round(sum(_shannon_entropy(v) for v in values) / len(values), 3)


def extract_features(session_id: str, conn: sqlite3.Connection = None) -> dict:
    """
    Returns a dict with every key in FEATURE_COLUMNS, plus 'session_id',
    'source_ip', 'protocol' for reference (not fed to the model).
    Raises ValueError if the session_id has no logged events at all.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()

    rows = conn.execute(
        "SELECT * FROM attack_logs WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()

    if not rows:
        if own_conn:
            conn.close()
        raise ValueError(f"No logged events found for session_id={session_id}")

    meta = conn.execute(
        "SELECT source_ip, protocol, start_time FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    source_ip = meta["source_ip"] if meta else rows[0]["source_ip"]
    protocol = meta["protocol"] if meta else rows[0]["protocol"]
    start_time_str = meta["start_time"] if meta else rows[0]["timestamp"]

    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    start_time = datetime.fromisoformat(start_time_str)
    duration_seconds = (timestamps[-1] - start_time).total_seconds()

    gaps = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    avg_seconds_between = sum(gaps) / len(gaps) if gaps else 0.0

    usernames = {r["username"] for r in rows if r["username"]}
    passwords = {r["password"] for r in rows if r["password"]}
    auth_attempts = sum(1 for r in rows if r["event_type"] == "auth_attempt")
    command_count = sum(1 for r in rows if r["event_type"] in ("command", "probe"))


    repeated_password_ratio = (
        round(1 - (len(passwords) / auth_attempts), 3)
        if auth_attempts > 0 else 0.0
    )
    repeated_password_ratio = max(0.0, min(1.0, repeated_password_ratio))

    payloads = [r["payload"] for r in rows if r["payload"]]
    unique_payloads = len(set(payloads))
    avg_payload_length = (
        round(sum(len(p) for p in payloads) / len(payloads), 2) if payloads else 0.0
    )

    effective_duration = max(duration_seconds, 1.0)  # avoid divide-by-near-zero blowups
    commands_per_minute = round(command_count / (effective_duration / 60), 3)

    recent_count = conn.execute(
        """SELECT COUNT(*) FROM sessions
           WHERE source_ip = ? AND session_id != ?
             AND (JULIANDAY(?) - JULIANDAY(start_time)) * 86400 <= ?
             AND start_time <= ?""",
        (source_ip, session_id, start_time_str, RECENT_WINDOW_SECONDS, start_time_str),
    ).fetchone()[0]

    features = {
        "event_count": len(rows),
        "auth_attempts": auth_attempts,
        "unique_usernames": len(usernames),
        "unique_passwords": len(passwords),
        "command_count": command_count,
        "duration_seconds": round(duration_seconds, 3),
        "avg_seconds_between": round(avg_seconds_between, 3),
        "recent_session_count_from_ip": recent_count,
        "username_entropy": _avg_entropy(usernames),
        "password_entropy": _avg_entropy(passwords),
        "repeated_password_ratio": repeated_password_ratio,
        "unique_payloads": unique_payloads,
        "avg_payload_length": avg_payload_length,
        "commands_per_minute": commands_per_minute,
    }
    for p in PROTOCOLS:
        features[f"is_{p.lower()}"] = 1 if protocol == p else 0

    if own_conn:
        conn.close()

    features["session_id"] = session_id
    features["source_ip"] = source_ip
    features["protocol"] = protocol
    return features


def features_to_vector(features: dict) -> list:
    """Extract just the model-ready numeric values, in FEATURE_COLUMNS order."""
    return [features[col] for col in FEATURE_COLUMNS]


if __name__ == "__main__":
    # Quick manual test: python3 ml/feature_extraction.py <session_id>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ml/feature_extraction.py <session_id>")
        sys.exit(1)
    f = extract_features(sys.argv[1])
    for k, v in f.items():
        print(f"  {k}: {v}")