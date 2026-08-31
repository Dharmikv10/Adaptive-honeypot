"""
run.py

Phase 2 launcher: runs all five decoy/dashboard components on a single
asyncio event loop.

  - fake_ftp, fake_telnet, fake_db, dashboard  -> true asyncio coroutines
  - fake_http                                   -> asyncio coroutine (uvicorn
                                                    supports this natively)
  - fake_ssh                                    -> paramiko is blocking, so
                                                    it runs in a background
                                                    thread via asyncio.to_thread

Usage:
    source venv/bin/activate
    python3 run.py
Ctrl+C to stop everything.
"""

import asyncio

from db.database import init_db
from services import fake_ssh, fake_http, fake_ftp, fake_telnet, fake_db
from dashboard import app as dashboard


async def main():
    init_db()

    print("[run] Adaptive Honeypot Phase 2 starting...")
    print(f"[run]   Fake SSH     -> port {fake_ssh.LISTEN_PORT}")
    print(f"[run]   Fake HTTP    -> port {fake_http.LISTEN_PORT}")
    print(f"[run]   Fake FTP     -> port {fake_ftp.LISTEN_PORT}")
    print(f"[run]   Fake Telnet  -> port {fake_telnet.LISTEN_PORT}")
    print(f"[run]   Fake MySQL   -> port {fake_db.LISTEN_PORT}")
    print(f"[run]   Dashboard    -> http://0.0.0.0:{dashboard.LISTEN_PORT}")
    print("[run] Press Ctrl+C to stop.")

    await asyncio.gather(
        asyncio.to_thread(fake_ssh.run),   # blocking paramiko loop -> its own thread
        fake_http.serve(),
        fake_ftp.serve(),
        fake_telnet.serve(),
        fake_db.serve(),
        dashboard.serve(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[run] Shutting down.")