"""
services/fake_http.py

Fake HTTP service mimicking a vulnerable CMS admin login page.
Logs every login attempt (username/password) and every request path hit
(useful later for spotting recon/scanning behavior), then always shows
an "invalid credentials" response. No real authentication exists.

Session tracking: HTTP is stateless (no persistent connection like SSH/FTP),
so a "session" here means all requests from one IP within a short window
(SESSION_TIMEOUT seconds) -- this approximates a single attacker visit/tool
run for feature_extraction.py, without needing cookies or auth state.

Run standalone for testing:
    python3 services/fake_http.py
(serves on port 8080 by default)
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import init_db, log_attempt_async, start_session, end_session_async, get_policy_async
from ml.adaption import schedule_session_processing

LISTEN_PORT = 8080
SESSION_TIMEOUT = 60  # seconds of inactivity before a new "session" starts for that IP

app = FastAPI()

# In-memory map of active HTTP "sessions" by source IP. Single-threaded asyncio
# event loop means no lock is needed here -- there's no await between the
# dict read and write in get_session_id().
_active_sessions: dict[str, dict] = {}

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Admin Login - CMS Panel</title></head>
<body style="font-family: sans-serif; max-width: 400px; margin: 100px auto;">
    <h2>CMS Administration</h2>
    {error}
    <form method="post" action="/login">
        <p><label>Username</label><br>
        <input type="text" name="username" style="width:100%"></p>
        <p><label>Password</label><br>
        <input type="password" name="password" style="width:100%"></p>
        <p><button type="submit">Log In</button></p>
    </form>
</body>
</html>
"""
DECOY_EXTRA_LINKS = """
<p style="margin-top:2em;font-size:0.85em;color:#888;">
<a href="/backup.zip">backup.zip</a> ·
<a href="/wp-admin">wp-admin</a> ·
<a href="/phpmyadmin">phpMyAdmin</a> ·
<a href="/.env">.env</a>
</p>
"""


@app.middleware("http")
async def apply_adaptive_policy(request: Request, call_next):
    """
    Phase 4: consults ip_policy before/after every request so HTTP behaves
    adaptively too, not just SSH -- tarpit delay, banner spoofing, and
    escalated logging of the User-Agent.
    """
    ip = client_ip_from(request)
    policy = await get_policy_async(ip)
    request.state.policy = policy

    if policy.get("tarpit"):
        await asyncio.sleep(random.uniform(2, 5))

    response = await call_next(request)

    if policy.get("banner_override"):
        try:
            server_banner = json.loads(policy["banner_override"]).get("HTTP")
            if server_banner:
                response.headers["server"] = server_banner
        except (TypeError, ValueError):
            pass

    return response

def client_ip_from(request: Request) -> str:
    # Behind NAT this is fine; if you ever put this behind a reverse proxy,
    # you'd read X-Forwarded-For instead (with the usual spoofing caveats).
    return request.client.host


async def get_session_id(client_ip: str) -> str:
    """Return the current session_id for this IP, starting a new one if the
    previous one has gone idle for longer than SESSION_TIMEOUT."""
    now = time.time()
    entry = _active_sessions.get(client_ip)
    if entry and (now - entry["last_seen"]) < SESSION_TIMEOUT:
        entry["last_seen"] = now
        return entry["session_id"]

    if entry:
        await end_session_async(entry["session_id"])
        schedule_session_processing(entry["session_id"])

    session_id = await asyncio.to_thread(start_session, client_ip, "HTTP", dest_port=LISTEN_PORT)
    _active_sessions[client_ip] = {"session_id": session_id, "last_seen": now}
    return session_id


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ip = client_ip_from(request)
    session_id = await get_session_id(ip)
    await log_attempt_async(
        source_ip=ip, protocol="HTTP", dest_port=LISTEN_PORT,
        event_type="request", payload="GET /", session_id=session_id,
    )
    page = LOGIN_PAGE.format(error="")
    if request.state.policy.get("decoy_expanded"):
        page += DECOY_EXTRA_LINKS
    return page


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    ip = client_ip_from(request)
    session_id = await get_session_id(ip)
    await log_attempt_async(
        source_ip=ip, protocol="HTTP", dest_port=LISTEN_PORT,
        event_type="request", payload="GET /login", session_id=session_id,
    )
    return LOGIN_PAGE.format(error="")


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = client_ip_from(request)
    session_id = await get_session_id(ip)
    notes = None
    if request.state.policy.get("escalated_logging"):
        notes = f"user_agent={request.headers.get('user-agent', 'unknown')}"
    await log_attempt_async(
        source_ip=ip, protocol="HTTP", dest_port=LISTEN_PORT,
        event_type="auth_attempt", username=username, password=password,
        payload="POST /login", notes=notes, session_id=session_id,
    )
    error_html = '<p style="color:red;">Invalid username or password.</p>'
    page = LOGIN_PAGE.format(error=error_html)
    if request.state.policy.get("decoy_expanded"):
        page += DECOY_EXTRA_LINKS
    return page


# Catch-all: logs any other path an attacker probes (e.g. /wp-admin, /.env, /phpmyadmin)
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(full_path: str, request: Request):
    ip = client_ip_from(request)
    session_id = await get_session_id(ip)
    await log_attempt_async(
        source_ip=ip, protocol="HTTP", dest_port=LISTEN_PORT,
        event_type="probe", payload=f"{request.method} /{full_path}",
        session_id=session_id,
    )
    return HTMLResponse("<h3>404 Not Found</h3>", status_code=404)


async def _reap_idle_sessions():
    """
    Background loop: since an HTTP 'session' only closes when a *new* request
    arrives after SESSION_TIMEOUT, an attacker who goes quiet (or never comes
    back) would leave their session stuck open forever. This closes anything
    idle for longer than SESSION_TIMEOUT so the sessions table stays accurate
    even with no further traffic -- important for the dashboard and ML layer,
    which both read finalized session rows.
    """
    while True:
        await asyncio.sleep(15)
        now = time.time()
        stale_ips = [
            ip for ip, entry in _active_sessions.items()
            if (now - entry["last_seen"]) >= SESSION_TIMEOUT
        ]
        for ip in stale_ips:
            entry = _active_sessions.pop(ip)
            await end_session_async(entry["session_id"])
            schedule_session_processing(entry["session_id"])


async def serve():
    """
    Asyncio-native entry point -- used by run.py so fake_http shares the same
    event loop as fake_ftp/fake_telnet/fake_db instead of needing its own thread.
    uvicorn.Server().serve() is itself a coroutine, so this integrates cleanly.
    """
    asyncio.create_task(_reap_idle_sessions())
    config = uvicorn.Config(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
    server = uvicorn.Server(config)
    print(f"[fake_http] Listening on port {LISTEN_PORT} (decoy)...")
    await server.serve()


def run():
    """Standalone entry point for testing this service by itself."""
    init_db()
    asyncio.run(serve())


if __name__ == "__main__":
    run()