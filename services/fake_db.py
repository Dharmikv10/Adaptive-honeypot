"""
services/fake_db.py

Fake database decoy, native asyncio. Sends a real MySQL protocol
"initial handshake" packet on connect -- enough that nmap's service
detection, mysql clients, and most scanners will fingerprint this port
as an actual MySQL server. Logs the connection and any raw bytes the
client sends back (a real login attempt packet, sqlmap probing, etc.)
before closing. We don't implement the full MySQL auth handshake --
there's no real database here to protect, only signal to capture.

Run standalone for testing:
    python3 services/fake_db.py
"""

import asyncio
import json
import random
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import init_db, log_attempt_async, start_session, end_session_async, get_policy_async
from ml.adaption import schedule_session_processing

LISTEN_PORT = 3307          # decoy port -- kept off the real MySQL port 3306
SERVER_VERSION = b"8.0.34-0ubuntu0.22.04.1"


def build_mysql_handshake_packet(server_version: bytes = SERVER_VERSION) -> bytes:
    """
    Minimal but structurally valid MySQL protocol v10 handshake packet.
    Real field-by-field spec: protocol version, server version (null-terminated),
    thread id, 8-byte auth-plugin-data part 1, filler, capability flags, etc.
    We only need enough for clients/scanners to recognize "this is MySQL" --
    we never complete real authentication.
    """
    protocol_version = b"\x0a"
    thread_id = struct.pack("<I", 1234)
    auth_data_1 = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    filler = b"\x00"
    capability_flags_lower = b"\xff\xf7"
    charset = b"\x08"
    status_flags = b"\x02\x00"
    capability_flags_upper = b"\xff\x81"
    auth_plugin_len = b"\x15"
    reserved = b"\x00" * 10
    auth_data_2 = b"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x00"
    auth_plugin_name = b"mysql_native_password\x00"

    payload = (
        protocol_version
        + server_version + b"\x00"
        + thread_id
        + auth_data_1
        + filler
        + capability_flags_lower
        + charset
        + status_flags
        + capability_flags_upper
        + auth_plugin_len
        + reserved
        + auth_data_2
        + auth_plugin_name
    )
    length = struct.pack("<I", len(payload))[:3]
    sequence_id = b"\x00"
    return length + sequence_id + payload


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    client_ip = peer[0] if peer else "unknown"

    policy = await get_policy_async(client_ip)
    if policy.get("tarpit"):
        await asyncio.sleep(random.uniform(3, 7))

    version = SERVER_VERSION.decode()
    if policy.get("banner_override"):
        try:
            version = json.loads(policy["banner_override"]).get("MYSQL", version)
        except (TypeError, ValueError):
            pass
    escalated = bool(policy.get("escalated_logging"))

    session_id = start_session(client_ip, "MYSQL", dest_port=LISTEN_PORT)
    await log_attempt_async(source_ip=client_ip, protocol="MYSQL", dest_port=LISTEN_PORT,
                             event_type="connection", session_id=session_id,
                             notes="tarpitted" if policy.get("tarpit") else None)

    try:
        writer.write(build_mysql_handshake_packet(version.encode()))
        await writer.drain()

        # Whatever comes back is the client's login-request packet (or a
        # scanner/exploit tool probing). We can't parse full MySQL auth here,
        # but the raw bytes are still useful forensic signal.
        data = await asyncio.wait_for(reader.read(512), timeout=10)
        if data:
            notes = f"{len(data)} bytes received after handshake"
            if escalated:
                notes += " [escalated_logging]"
            await log_attempt_async(
                source_ip=client_ip, protocol="MYSQL", dest_port=LISTEN_PORT,
                event_type="probe", payload=data.hex(),
                notes=notes,
                session_id=session_id,
            )
    except (asyncio.TimeoutError, ConnectionResetError):
        pass
    except Exception as e:
        await log_attempt_async(source_ip=client_ip, protocol="MYSQL", dest_port=LISTEN_PORT,
                                 event_type="error", notes=str(e), session_id=session_id)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        await end_session_async(session_id)
        schedule_session_processing(session_id)


async def serve():
    server = await asyncio.start_server(handle_client, "0.0.0.0", LISTEN_PORT)
    print(f"[fake_db] Listening on port {LISTEN_PORT} (decoy, MySQL-style)...")
    async with server:
        await server.serve_forever()


def run():
    init_db()
    asyncio.run(serve())


if __name__ == "__main__":
    run()