"""
ml/adaptation.py

The Adaptive Policy Decision Engine. This is what makes the honeypot
"adaptive" rather than just a logger with a classifier bolted on: it takes
a session's ML prediction and decides what the honeypot should actually DO
about it, then writes that decision into ip_policy so every other service
can act on it going forward.

Mapping (matches the roadmap's Phase 4 examples):
    Recon Scan           -> expand deception surface (richer decoy content)
    Brute Force           -> tarpit (slow every response for that IP)
    Credential Stuffing    -> tarpit + escalated logging
    Web Enumeration        -> expand deception surface + escalated logging
    Exploit Attempt        -> escalated logging (deep logging)
    Malware Download       -> escalated logging + packet capture
    Unknown Behaviour      -> escalated logging only (learn more, act less)
    Repeat offender (any class, independent trigger) -> packet capture

Threat scoring: a simple cumulative severity score per IP, incremented every
time that IP gets classified, weighted by how dangerous the category is and
boosted for repeat visits. This is what the Phase 5 dashboard's "Threat
Level" widget will read from ip_policy.threat_score.

Usage:
    from ml.adaptation import decide_and_apply
    decision = decide_and_apply(session_id)
    # decision = {"predicted_class": ..., "confidence": ..., "actions": [...],
    #             "threat_score": ..., "reason": ...}

Called automatically by every service after a session closes (see run.py /
each fake_*.py's finally block) -- you don't normally call this by hand,
though ml/predict.py-style standalone testing still works via __main__.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import get_policy, update_policy, record_session_intelligence
from ml.predict import predict_session
from ml.packet_capture import start_capture_for_ip

# Severity weight per class -- used to build the cumulative threat_score.
# Base severity per class, already on the final 0-10 scale.
SEVERITY = {
    "Recon Scan": 2.0,
    "Web Enumeration": 4.0,
    "Brute Force": 6.0,
    "Credential Stuffing": 7.0,
    "Exploit Attempt": 9.0,
    "Malware Download": 10.0,
    "Unknown Behaviour": 2.0,
}

MAX_THREAT_SCORE = 10.0
REPEAT_OFFENDER_BONUS = 2.0

# Alternate banners the decision engine can recommend rotating an IP onto
# once it's a confirmed repeat offender -- makes fingerprinting tools less
# reliable against us. Picked deterministically per-IP so the same offender
# doesn't flip banners every single session.
# NEW — one pool per protocol, since an IP can hit multiple decoy
# services and each needs a protocol-appropriate banner.
BANNER_POOL = {
    "SSH": [
        "SSH-2.0-OpenSSH_8.9p1",
        "SSH-2.0-OpenSSH_9.6",
        "SSH-2.0-dropbear_2022.83",
    ],
    "FTP": [
        "220 (vsFTPd 3.0.5)\r\n",
        "220 ProFTPD 1.3.8 Server ready.\r\n",
        "220 Pure-FTPd [privsep]\r\n",
    ],
    "TELNET": [
        "\r\nDebian GNU/Linux 11\r\n",
        "\r\nUbuntu 20.04.6 LTS\r\n",
        "\r\nCentOS Linux 7\r\n",
    ],
    "MYSQL": [
        "8.0.34-0ubuntu0.22.04.1",
        "8.0.36",
        "5.7.42-log",
    ],
    "HTTP": [
        "Apache/2.4.52 (Ubuntu)",
        "nginx/1.18.0",
        "Apache/2.4.41 (Ubuntu)",
    ],
}

# Repeat-offender thresholds: either signal is enough to trigger packet capture.
REPEAT_TIMES_FLAGGED_THRESHOLD = 3
REPEAT_RECENT_SESSIONS_THRESHOLD = 4

# Confidence floor: below this, we still log the prediction but don't act on
# it -- an uncertain guess shouldn't trigger tarpitting or logging escalation.
CONFIDENCE_ACTION_THRESHOLD = 0.55

def compute_threat_score(predicted_class: str, confidence: float, is_repeat_offender: bool) -> float:
    """
    Bounded 0-10 Threat Score for a single session:
        threat_score = min(10, SEVERITY[predicted_class] * confidence
                                + (REPEAT_OFFENDER_BONUS if is_repeat_offender else 0))
    """
    severity = SEVERITY.get(predicted_class, 2.0)
    raw = severity * confidence + (REPEAT_OFFENDER_BONUS if is_repeat_offender else 0.0)
    return round(min(MAX_THREAT_SCORE, max(0.0, raw)), 1)


def _rules_for_class(predicted_class: str) -> set:
    """Returns the set of action names this classification alone would trigger."""
    rules = {
        "Recon Scan": {"expand_decoy", "connection_delay"},
        "Brute Force": {"tarpit"},
        "Credential Stuffing": {"tarpit", "escalate_logging"},
        "Web Enumeration": {"expand_decoy", "escalate_logging", "connection_delay"},
        "Exploit Attempt": {"escalate_logging"},
        "Malware Download": {"escalate_logging", "packet_capture"},
        "Unknown Behaviour": {"escalate_logging"},
    }
    return rules.get(predicted_class, set())


def decide_and_apply(session_id: str) -> dict:
    """
    Runs classification (Phase 3) + policy decision (Phase 4) for one session,
    applies the resulting policy to ip_policy, and records everything onto
    the session row. Safe to call even for very short/low-signal sessions --
    predict_session() will just return a low-confidence guess in that case.
    """
    prediction = predict_session(session_id)
    predicted_class = prediction["predicted_class"]
    confidence = prediction["confidence"]

    # predict_session's top-level dict doesn't include source_ip directly
    # (it's stripped out of the scored feature vector) -- pull it from the
    # session row instead.
# NEW
    from db.database import DB_PATH
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT source_ip, protocol FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    source_ip = row[0] if row else "unknown"
    protocol = (row[1] if row else "SSH") or "SSH"

    policy = get_policy(source_ip)
    times_flagged = policy["times_flagged"] + 1
    is_repeat_offender = (
        times_flagged >= REPEAT_TIMES_FLAGGED_THRESHOLD
        or prediction["features"].get("recent_session_count_from_ip", 0) >= REPEAT_RECENT_SESSIONS_THRESHOLD
    )

    actions = set()
    if confidence >= CONFIDENCE_ACTION_THRESHOLD:
        actions |= _rules_for_class(predicted_class)
    if is_repeat_offender:
        actions.add("packet_capture")
        actions.add("banner_override")

    # Threat score: severity of this classification, discounted by confidence
    # (an uncertain guess contributes less), plus a flat bump for repeat visits.
    # Threat score: bounded 0-10, computed fresh for THIS session (not
    # cumulative -- each session gets its own rating).
    threat_score = compute_threat_score(predicted_class, confidence, is_repeat_offender)

    # ip_policy.threat_score tracks this IP's PEAK score so far (still 0-10,
    # never decreases) -- used for IP-level policy, not for dashboard display.
    policy_update = {
        "threat_score": round(max(policy["threat_score"], threat_score), 1),
        "times_flagged": times_flagged,
        "last_classification": predicted_class,
    }
    if "tarpit" in actions:
        policy_update["tarpit"] = 1
    if "escalate_logging" in actions:
        policy_update["escalated_logging"] = 1
    if "expand_decoy" in actions:
        policy_update["decoy_expanded"] = 1
    new_banner_for_reason = None
    if "banner_override" in actions:
        # Deterministic per-IP/per-protocol pick (not random) so the same
        # offender doesn't flip banners every session. Stored as a JSON map
        # keyed by protocol (e.g. {"SSH": "...", "FTP": "..."}) so each
        # service only ever reads/overrides its own key -- one IP hitting
        # multiple decoy services doesn't clobber the others' banners.
        pool = BANNER_POOL.get(protocol, BANNER_POOL["SSH"])
        try:
            overrides = json.loads(policy.get("banner_override") or "{}")
        except (TypeError, ValueError):
            overrides = {}
        new_banner_for_reason = pool[times_flagged % len(pool)]
        overrides[protocol] = new_banner_for_reason
        policy_update["banner_override"] = json.dumps(overrides)

    update_policy(source_ip, **policy_update)

    if "packet_capture" in actions:
        start_capture_for_ip(source_ip)

    reason = prediction["reason"]
    if is_repeat_offender:
        reason += (f" Also flagged as a repeat offender ({times_flagged} sessions from this IP); "
                    f"recommending banner rotation to '{new_banner_for_reason}' for {protocol}.")

    action_summary = ", ".join(sorted(actions)) if actions else "none (below confidence threshold)"

    record_session_intelligence(
        session_id=session_id,
        predicted_class=predicted_class,
        confidence=confidence,
        reason=reason,
        adaptation_taken=action_summary,
        threat_score=threat_score,
    )

    result = {
        "session_id": session_id,
        "source_ip": source_ip,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "reason": reason,
        "actions": sorted(actions),
        "threat_score": threat_score,
        "ip_peak_threat_score": policy_update["threat_score"],
        "is_repeat_offender": is_repeat_offender,
    }
    print(f"[adaptation] {source_ip} -> {predicted_class} ({confidence:.2f}) "
          f"actions=[{action_summary}] threat_score={threat_score}/10")
    return result


async def process_session_async(session_id: str):
    """
    Fire-and-forget entry point used by the asyncio services: classification
    + adaptation shouldn't block tearing down the connection. Swallows
    ValueError from feature_extraction (empty session -- e.g. a bare TCP
    connect/disconnect with zero logged events) since there's nothing to
    classify in that case.
    """
    try:
        return await asyncio.to_thread(decide_and_apply, session_id)
    except ValueError as e:
        print(f"[adaptation] skipped {session_id}: {e}")
        return None
    except Exception as e:
        print(f"[adaptation] error processing {session_id}: {e}")
        return None


def process_session_sync(session_id: str):
    """
    Synchronous counterpart to process_session_async, for services that
    already run their connection handler in a dedicated background thread
    rather than on the asyncio event loop (currently just fake_ssh, since
    paramiko is blocking). Same error-swallowing behavior as the async
    version -- an empty/low-signal session just gets skipped.
    """
    try:
        return decide_and_apply(session_id)
    except ValueError as e:
        print(f"[adaptation] skipped {session_id}: {e}")
        return None
    except Exception as e:
        print(f"[adaptation] error processing {session_id}: {e}")
        return None


# Registry of in-flight fire-and-forget tasks. asyncio.create_task() only
# holds a *weak* reference to the task via the event loop -- if nothing else
# references it, the task can be garbage-collected before it finishes
# running, silently dropping the classification. Keeping a strong reference
# here (and clearing it via the done-callback once the task finishes) avoids
# that footgun. See: https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_background_tasks: set = set()


def schedule_session_processing(session_id: str):
    """
    Fire-and-forget scheduler for the asyncio-based services (fake_ftp,
    fake_http, fake_telnet, fake_db): kicks off classification + adaptation
    as a background task so it never blocks connection teardown, while still
    guarding against the task being garbage-collected mid-flight.

    Call this right after end_session_async(session_id) in each service's
    'finally' block (or wherever a session is finalized, e.g. fake_http's
    idle-session reaper).
    """
    task = asyncio.create_task(process_session_async(session_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ml/adaptation.py <session_id>")
        sys.exit(1)
    result = decide_and_apply(sys.argv[1])
    print(json.dumps(result, indent=2))