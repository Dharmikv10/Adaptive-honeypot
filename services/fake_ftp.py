"""
services/fake_ftp.py

Fake FTP server, native asyncio (asyncio.start_server). Speaks just enough
real FTP protocol to look legitimate to a scanner or a human trying to log
in: greets with a banner, accepts USER/PASS, logs whatever credentials are
tried, always denies login, then logs a handful of post-auth commands
(LIST, CWD, etc.) before disconnecting -- useful signal even though the
attacker never actually gets in.

Run standalone for testing:
    python3 services/fake_ftp.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import init_db, log_attempt_async, start_session, end_session_async, get_policy_async
from ml.adaption import schedule_session_processing

LISTEN_PORT = 2121
BANNER = "220 (vsFTPd 3.0.3)\r\n"          # fake banner; Phase 4 makes this spoofable per-attacker
MAX_COMMANDS_LOGGED = 10                    # cap so a malicious client can't spam us forever


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    client_ip = peer[0] if peer else "unknown"

    policy = await get_policy_async(client_ip)
    if policy.get("tarpit"):
        await asyncio.sleep(random.uniform(3, 7))

    active_banner = BANNER
    if policy.get("banner_override"):
        try:
            active_banner = json.loads(policy["banner_override"]).get("FTP", BANNER)
        except (TypeError, ValueError):
            pass
    escalated = bool(policy.get("escalated_logging"))

    # One session per connection -- every event below is tagged with this id
    # so feature_extraction.py can later pull the whole session as one unit.
    session_id = start_session(client_ip, "FTP", dest_port=LISTEN_PORT)
    await log_attempt_async(source_ip=client_ip, protocol="FTP", dest_port=LISTEN_PORT,
                             event_type="connection", session_id=session_id,
                             notes="tarpitted" if policy.get("tarpit") else None)

    username = None
    try:
        writer.write(active_banner.encode())
        await writer.drain()

        commands_seen = 0
        while commands_seen < MAX_COMMANDS_LOGGED:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            commands_seen += 1

            verb, _, arg = text.partition(" ")
            verb = verb.upper()

            if verb == "USER":
                username = arg
                writer.write(b"331 Please specify the password.\r\n")
                
            elif verb == "PASS":
                await log_attempt_async(
                    source_ip=client_ip, protocol="FTP", dest_port=LISTEN_PORT,
                    event_type="auth_attempt", username=username, password=arg,
                    notes="escalated_logging" if escalated else None,
                    session_id=session_id,
                )
                writer.write(b"530 Login incorrect.\r\n")
            elif verb == "QUIT":
                writer.write(b"221 Goodbye.\r\n")
                await writer.drain()
                break
            else:
                # anything tried after a failed login is still useful signal
                await log_attempt_async(
                    source_ip=client_ip, protocol="FTP", dest_port=LISTEN_PORT,
                    event_type="command", username=username, payload=text,
                    session_id=session_id,
                )
                writer.write(b"530 Please login with USER and PASS.\r\n")

            await writer.drain()

    except (asyncio.TimeoutError, ConnectionResetError):
        pass
    except Exception as e:
        await log_attempt_async(source_ip=client_ip, protocol="FTP", dest_port=LISTEN_PORT,
                                 event_type="error", notes=str(e), session_id=session_id)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        await end_session_async(session_id)
        # Kick off classification + adaptation as a background task -- don't
        # await it here, we don't want a slow ML call to delay accepting the
        # next connection.
        schedule_session_processing(session_id)


async def serve():
    """Start the asyncio server and run forever. Awaitable -- used by run.py."""
    server = await asyncio.start_server(handle_client, "0.0.0.0", LISTEN_PORT)
    print(f"[fake_ftp] Listening on port {LISTEN_PORT} (decoy)...")
    async with server:
        await server.serve_forever()


def run():
    """Standalone entry point for testing this service by itself."""
    init_db()
    asyncio.run(serve())


if __name__ == "__main__":
    run()