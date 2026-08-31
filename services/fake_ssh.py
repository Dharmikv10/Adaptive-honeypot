"""
services/fake_ssh.py

Fake SSH server built on paramiko. Accepts TCP connections on a configurable
port, performs a real SSH handshake, then logs every username/password the
attacker tries and DENIES all authentication (no shell is ever granted).

Run standalone for testing:
    python3 services/fake_ssh.py
"""
import json
import socket
import threading
import time
import sys
from pathlib import Path

import paramiko

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import init_db, log_attempt, start_session, end_session, get_policy
from ml.adaption import process_session_sync

LISTEN_PORT = 2223
BANNER = "SSH-2.0-OpenSSH_7.4"
HOST_KEY_PATH = Path(__file__).parent / "fake_ssh_host_key"
TARPIT_DELAY_SECONDS = (3, 7)


def get_or_create_host_key():
    """Generate a persistent RSA host key on first run so the fingerprint stays stable."""
    if HOST_KEY_PATH.exists():
        return paramiko.RSAKey(filename=str(HOST_KEY_PATH))
    key = paramiko.RSAKey.generate(2048)
    key.write_private_key_file(str(HOST_KEY_PATH))
    return key


class FakeSSHServer(paramiko.ServerInterface):
    """
    paramiko calls these methods during the SSH handshake.
    We log everything and always reject, so no attacker ever gets a shell.
    """

    def __init__(self, client_ip, session_id, escalated_logging=False):
        self.client_ip = client_ip
        self.session_id = session_id
        self.escalated_logging = escalated_logging
        self.event = threading.Event()

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED

    def check_auth_password(self, username, password):
        notes = f"client_version={self.transport.remote_version}" \
            if self.escalated_logging and getattr(self, "transport", None) else None
        log_attempt(
            source_ip=self.client_ip,
            protocol="SSH",
            dest_port=LISTEN_PORT,
            event_type="auth_attempt",
            username=username,
            password=password,
            notes=notes,
            session_id=self.session_id,
        )
        return paramiko.AUTH_FAILED

    def check_auth_publickey(self, username, key):
        log_attempt(
            source_ip=self.client_ip,
            protocol="SSH",
            dest_port=LISTEN_PORT,
            event_type="auth_attempt_pubkey",
            username=username,
            payload=f"key_fingerprint={key.get_fingerprint().hex()}",
            session_id=self.session_id,
        )
        return paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "password,publickey"

    def check_channel_shell_request(self, channel):
        return False

    def check_channel_pty_request(self, *args, **kwargs):
        return False


def handle_connection(client_socket, client_addr):
    import random

    client_ip = client_addr[0]
    policy = get_policy(client_ip)

    if policy.get("tarpit"):
        time.sleep(random.uniform(*TARPIT_DELAY_SECONDS))

    session_id = start_session(client_ip, "SSH", dest_port=LISTEN_PORT)
    log_attempt(source_ip=client_ip, protocol="SSH", dest_port=LISTEN_PORT,
                event_type="connection", session_id=session_id,
                notes="tarpitted" if policy.get("tarpit") else None)

    active_banner = BANNER
    if policy.get("banner_override"):
        try:
            active_banner = json.loads(policy["banner_override"]).get("SSH", BANNER)
        except (TypeError, ValueError):
            pass

    transport = None
    try:
        transport = paramiko.Transport(client_socket)
        transport.local_version = active_banner
        transport.add_server_key(get_or_create_host_key())
        server = FakeSSHServer(client_ip, session_id, escalated_logging=bool(policy.get("escalated_logging")))
        server.transport = transport
        transport.start_server(server=server)

        # Wait briefly for an auth attempt; we already logged it in check_auth_password.
        # Since auth always fails, the attacker's client will retry or disconnect on its own.
        chan = transport.accept(timeout=10)
        if chan is not None:
            chan.close()
    except Exception as e:
        log_attempt(source_ip=client_ip, protocol="SSH", dest_port=LISTEN_PORT,
                    event_type="error", notes=str(e), session_id=session_id)
    finally:
        if transport is not None:
            transport.close()
        end_session(session_id)
        # handle_connection already runs in its own daemon thread (see run()
        # below), so it's safe to classify + adapt synchronously here without
        # blocking the accept loop or any other connection.
        process_session_sync(session_id)


def run():
    init_db()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", LISTEN_PORT))
    sock.listen(100)
    print(f"[fake_ssh] Listening on port {LISTEN_PORT} (decoy)...")

    while True:
        client_socket, client_addr = sock.accept()
        thread = threading.Thread(
            target=handle_connection, args=(client_socket, client_addr), daemon=True
        )
        thread.start()


if __name__ == "__main__":
    run()