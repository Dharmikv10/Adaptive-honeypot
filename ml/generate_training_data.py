"""
ml/generate_training_data.py

Bootstraps a labeled training set for the classifier. We don't have real
attack sessions yet -- that only happens in Phase 6, when the honeypot gets
attacked from the Kali VM -- so this generates synthetic feature vectors
with realistic distributions for each attack category, based on how each
attack type actually behaves against these decoy services:

  - Recon Scan        : many fast probe/command events, no real auth attempts
  - Brute Force        : many auth_attempts, few usernames, many passwords,
                          very short gaps (automated tool), one protocol
  - Credential Stuffing : like brute force, but many usernames AND passwords
                          (credential-list attacks), usually HTTP or SSH
  - Web Enumeration     : HTTP only, high command_count (many paths probed),
                          near-zero auth_attempts
  - Exploit Attempt     : short session, few auth attempts, some commands
                          (payload delivery), FTP/HTTP/MySQL
  - Malware Download    : very short session, minimal auth, 1-3 commands
                          (fetch + maybe execute), FTP/HTTP
  - Unknown Behaviour   : wide/overlapping random ranges -- a deliberate
                          noisy catch-all so the model has to learn there's
                          a "none of the above" option

IMPORTANT: this is clearly-labeled synthetic bootstrap data, not claimed as
real attack traffic. Phase 6 (real Kali VM attacks) should feed real labeled
sessions back into this training set -- see ml/train_model.py's note on
retraining with real data.

Run:
    python3 ml/generate_training_data.py
Writes: ml/data/synthetic_sessions.csv
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from feature_extraction import FEATURE_COLUMNS

OUT_PATH = Path(__file__).parent / "data" / "synthetic_sessions.csv"
SAMPLES_PER_CLASS = 400
RANDOM_SEED = 42

PROTOCOLS = ["SSH", "HTTP", "FTP", "TELNET", "MYSQL"]


def _protocol_row(protocol: str) -> dict:
    return {f"is_{p.lower()}": (1 if p == protocol else 0) for p in PROTOCOLS}


def _clip_nonneg(x):
    return max(0, x)


def gen_recon_scan(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(["HTTP", "MYSQL", "SSH", "FTP", "TELNET"], p=[0.5, 0.2, 0.1, 0.1, 0.1])
        event_count = int(rng.normal(12, 4)); event_count = max(3, event_count)
        row = {
            "event_count": event_count,
            "auth_attempts": 0,
            "unique_usernames": 0,
            "unique_passwords": 0,
            "command_count": _clip_nonneg(int(rng.normal(event_count * 0.8, 2))),
            "duration_seconds": round(abs(rng.normal(1.5, 1.0)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.1, 0.05)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(4, 2))),
            "username_entropy": 0.0,
            "password_entropy": 0.0,
            "repeated_password_ratio": 0.0,
            "unique_payloads": _clip_nonneg(int(rng.normal(event_count * 0.7, 2))),
            "avg_payload_length": round(abs(rng.normal(9, 3)), 2),
            "commands_per_minute": round(abs(rng.normal(220, 60)), 3),
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Recon Scan"
        rows.append(row)
    return rows


def gen_brute_force(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(["SSH", "FTP", "TELNET"], p=[0.5, 0.25, 0.25])
        auth_attempts = int(rng.normal(25, 8)); auth_attempts = max(5, auth_attempts)
        row = {
            "event_count": auth_attempts + int(rng.integers(0, 3)),
            "auth_attempts": auth_attempts,
            "unique_usernames": int(np.clip(rng.normal(1.5, 0.8), 1, 3)),
            "unique_passwords": int(np.clip(rng.normal(auth_attempts * 0.9, 3), 3, None)),
            "command_count": 0,
            "duration_seconds": round(abs(rng.normal(auth_attempts * 0.3, 2)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.3, 0.15)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(2, 1.5))),
            "username_entropy": round(abs(rng.normal(1.9, 0.3)), 3),
            "password_entropy": round(abs(rng.normal(2.3, 0.4)), 3),
            "repeated_password_ratio": round(np.clip(rng.normal(0.18, 0.08), 0, 1), 3),
            "unique_payloads": 0,
            "avg_payload_length": 0.0,
            "commands_per_minute": 0.0,
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Brute Force"
        rows.append(row)
    return rows


def gen_credential_stuffing(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(["HTTP", "SSH", "FTP"], p=[0.6, 0.2, 0.2])
        auth_attempts = int(rng.normal(30, 10)); auth_attempts = max(8, auth_attempts)
        row = {
            "event_count": auth_attempts + int(rng.integers(0, 3)),
            "auth_attempts": auth_attempts,
            "unique_usernames": int(np.clip(rng.normal(auth_attempts * 0.6, 4), 4, None)),
            "unique_passwords": int(np.clip(rng.normal(auth_attempts * 0.8, 4), 4, None)),
            "command_count": 0,
            "duration_seconds": round(abs(rng.normal(auth_attempts * 0.2, 2)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.2, 0.1)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(2, 1.5))),
            "username_entropy": round(abs(rng.normal(3.4, 0.3)), 3),
            "password_entropy": round(abs(rng.normal(3.6, 0.3)), 3),
            "repeated_password_ratio": round(np.clip(rng.normal(0.05, 0.04), 0, 1), 3),
            "unique_payloads": 0,
            "avg_payload_length": 0.0,
            "commands_per_minute": 0.0,
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Credential Stuffing"
        rows.append(row)
    return rows


def gen_web_enumeration(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        command_count = int(rng.normal(20, 6)); command_count = max(5, command_count)
        row = {
            "event_count": command_count + int(rng.integers(0, 2)),
            "auth_attempts": int(rng.integers(0, 2)),
            "unique_usernames": int(rng.integers(0, 2)),
            "unique_passwords": int(rng.integers(0, 2)),
            "command_count": command_count,
            "duration_seconds": round(abs(rng.normal(3, 1.5)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.15, 0.08)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(3, 2))),
            "username_entropy": 0.0,
            "password_entropy": 0.0,
            "repeated_password_ratio": 0.0,
            "unique_payloads": _clip_nonneg(int(rng.normal(command_count * 0.85, 3))),
            "avg_payload_length": round(abs(rng.normal(14, 4)), 2),
            "commands_per_minute": round(abs(rng.normal(260, 70)), 3),
        }
        row.update(_protocol_row("HTTP"))
        row["label"] = "Web Enumeration"
        rows.append(row)
    return rows


def gen_exploit_attempt(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(["HTTP", "MYSQL", "FTP"], p=[0.45, 0.35, 0.2])
        command_count = int(np.clip(rng.normal(4, 2), 1, None))
        row = {
            "event_count": command_count + int(rng.integers(1, 3)),
            "auth_attempts": int(rng.integers(0, 3)),
            "unique_usernames": int(rng.integers(0, 2)),
            "unique_passwords": int(rng.integers(0, 2)),
            "command_count": command_count,
            "duration_seconds": round(abs(rng.normal(2, 1.2)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.5, 0.3)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(1, 1))),
            "username_entropy": 0.0,
            "password_entropy": 0.0,
            "repeated_password_ratio": 0.0,
            "unique_payloads": max(1, int(np.clip(rng.normal(command_count * 0.7, 1), 1, None))),
            "avg_payload_length": round(abs(rng.normal(65, 20)), 2),
            "commands_per_minute": round(abs(rng.normal(40, 20)), 3),
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Exploit Attempt"
        rows.append(row)
    return rows


def gen_malware_download(rng):
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(["FTP", "HTTP"], p=[0.6, 0.4])
        command_count = int(np.clip(rng.normal(2, 1), 1, 4))
        row = {
            "event_count": command_count + int(rng.integers(1, 3)),
            "auth_attempts": int(rng.integers(0, 2)),
            "unique_usernames": int(rng.integers(0, 2)),
            "unique_passwords": int(rng.integers(0, 2)),
            "command_count": command_count,
            "duration_seconds": round(abs(rng.normal(1, 0.6)), 3),
            "avg_seconds_between": round(abs(rng.normal(0.4, 0.2)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(0.5, 0.8))),
            "username_entropy": 0.0,
            "password_entropy": 0.0,
            "repeated_password_ratio": 0.0,
            "unique_payloads": max(1, command_count),
            "avg_payload_length": round(abs(rng.normal(28, 8)), 2),
            "commands_per_minute": round(abs(rng.normal(90, 30)), 3),
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Malware Download"
        rows.append(row)
    return rows


def gen_unknown(rng):
    """Deliberately noisy / overlapping ranges -- teaches the model that not
    every session cleanly fits a named category."""
    rows = []
    for _ in range(SAMPLES_PER_CLASS):
        protocol = rng.choice(PROTOCOLS)
        row = {
            "event_count": int(abs(rng.normal(6, 5))) + 1,
            "auth_attempts": int(abs(rng.normal(1, 2))),
            "unique_usernames": int(abs(rng.normal(0.5, 1))),
            "unique_passwords": int(abs(rng.normal(0.5, 1))),
            "command_count": int(abs(rng.normal(2, 3))),
            "duration_seconds": round(abs(rng.normal(2, 3)), 3),
            "avg_seconds_between": round(abs(rng.normal(1, 1.5)), 3),
            "recent_session_count_from_ip": int(abs(rng.normal(1, 1.5))),
            "username_entropy": round(abs(rng.normal(1.2, 1.2)), 3),
            "password_entropy": round(abs(rng.normal(1.2, 1.2)), 3),
            "repeated_password_ratio": round(np.clip(rng.normal(0.3, 0.3), 0, 1), 3),
            "unique_payloads": int(abs(rng.normal(2, 2))),
            "avg_payload_length": round(abs(rng.normal(15, 15)), 2),
            "commands_per_minute": round(abs(rng.normal(30, 40)), 3),
        }
        row.update(_protocol_row(protocol))
        row["label"] = "Unknown Behaviour"
        rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    all_rows = []
    for gen_fn in [
        gen_recon_scan, gen_brute_force, gen_credential_stuffing,
        gen_web_enumeration, gen_exploit_attempt, gen_malware_download,
        gen_unknown,
    ]:
        all_rows.extend(gen_fn(rng))

    random.shuffle(all_rows)

    fieldnames = FEATURE_COLUMNS + ["label"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote {len(all_rows)} synthetic sessions to {OUT_PATH}")
    print(f"Classes: {SAMPLES_PER_CLASS} samples each x 7 classes")


if __name__ == "__main__":
    main()