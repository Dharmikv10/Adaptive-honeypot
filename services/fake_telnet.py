"""
services/fake_telnet.py

Fake Telnet server, native asyncio. Presents a plain login/password prompt
(the style most brute-force tools and IoT-targeting scripts expect), logs
every attempt, always rejects, then logs a few commands typed afterward
before disconnecting.

Run standalone for testing:
    python3 services/fake_telnet.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import init_db, log_attempt_async, start_session, end_session_async, get_policy_async
from ml.adaption import schedule_session_processing

LISTEN_PORT = 2323
BANNER = "\r\nUbuntu 22.04.5 LTS\r\n"
MAX_COMMANDS_LOGGED = 10


async def read_line(reader: asyncio.StreamReader, timeout: float = 30) -> str:
    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    return line.decode(errors="replace").strip()


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    client_ip = peer[0] if peer else "unknown"

    policy = await get_policy_async(client_ip)
    if policy.get("tarpit"):
        await asyncio.sleep(random.uniform(3, 7))

    active_banner = BANNER
    if policy.get("banner_override"):
        try:
            active_banner = json.loads(policy["banner_override"]).get("TELNET", BANNER)
        except (TypeError, ValueError):
            pass
    escalated = bool(policy.get("escalated_logging"))

    session_id = start_session(client_ip, "TELNET", dest_port=LISTEN_PORT)
    await log_attempt_async(source_ip=client_ip, protocol="TELNET", dest_port=LISTEN_PORT,
                             event_type="connection", session_id=session_id,
                             notes="tarpitted" if policy.get("tarpit") else None)

    try:
        writer.write(active_banner.encode())
        writer.write(b"login: ")
        await writer.drain()
        username = await read_line(reader)

        writer.write(b"Password: ")
        await writer.drain()
        password = await read_line(reader)

        await log_attempt_async(
            source_ip=client_ip, protocol="TELNET", dest_port=LISTEN_PORT,
            event_type="auth_attempt", username=username, password=password,
            notes="escalated_logging" if escalated else None,
            session_id=session_id,
        )

        writer.write(b"\r\nLogin incorrect\r\n")
        await writer.drain()

        # Some brute-force tools keep the socket open and retry, or a curious
        # human might type commands anyway -- log whatever comes next.
        commands_seen = 0
        while commands_seen < MAX_COMMANDS_LOGGED:
            cmd = await read_line(reader, timeout=15)
            if not cmd:
                break
            commands_seen += 1
            await log_attempt_async(
                source_ip=client_ip, protocol="TELNET", dest_port=LISTEN_PORT,
                event_type="command", username=username, payload=cmd,
                session_id=session_id,
            )
            writer.write(b"login: ")
            await writer.drain()

    except (asyncio.TimeoutError, ConnectionResetError):
        pass
    except Exception as e:
        await log_attempt_async(source_ip=client_ip, protocol="TELNET", dest_port=LISTEN_PORT,
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
    print(f"[fake_telnet] Listening on port {LISTEN_PORT} (decoy)...")
    async with server:
        await server.serve_forever()


def run():
    init_db()
    asyncio.run(serve())


if __name__ == "__main__":
    run()