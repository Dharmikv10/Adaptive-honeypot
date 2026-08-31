"""
ml/packet_capture.py

Deep packet capture for repeat offenders / high-severity classifications,
using Scapy's AsyncSniffer, as called for in the roadmap ("Repeat Offender
-> Packet Capture").

IMPORTANT PRIVILEGE NOTE: raw packet sniffing needs either root or the
CAP_NET_RAW/CAP_NET_ADMIN capabilities. Your Docker Compose setup drops ALL
capabilities on every container for isolation (see docker-compose.yml).
That means packet capture will silently fail with a PermissionError inside
containers unless you deliberately add those capabilities back for the
container(s) you want this to run in -- see the docker-compose.yml comment
next to `cap_add`. Outside Docker (running run.py directly on the VM), it
needs `sudo`.

Design choice: this fails soft. If we can't get a raw socket, we log a
warning and move on instead of crashing the honeypot -- a demo where every
OTHER feature keeps working is much better than one where a missing `sudo`
takes down the whole thing.
"""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CAPTURE_DIR = Path(__file__).resolve().parent.parent / "logs" / "pcap"
CAPTURE_DURATION_SECONDS = 60  # auto-stop after this long, so it can't run forever

_active_captures = {}  # source_ip -> sniffer object, so we don't double-start


def _run_capture(source_ip: str):
    try:
        from scapy.all import AsyncSniffer, wrpcap
    except ImportError:
        print("[packet_capture] scapy not installed -- skipping capture. "
              "Install with: pip install scapy")
        return

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = CAPTURE_DIR / f"{source_ip.replace(':', '_')}_{timestamp}.pcap"

    try:
        sniffer = AsyncSniffer(filter=f"host {source_ip}", store=True)
        sniffer.start()
        _active_captures[source_ip] = sniffer
        print(f"[packet_capture] started for {source_ip} -> {out_path}")

        time.sleep(CAPTURE_DURATION_SECONDS)

        packets = sniffer.stop()
        _active_captures.pop(source_ip, None)
        if packets:
            wrpcap(str(out_path), packets)
            print(f"[packet_capture] wrote {len(packets)} packets to {out_path}")
        else:
            print(f"[packet_capture] no packets captured for {source_ip} (nothing to write)")

    except PermissionError:
        print(f"[packet_capture] PERMISSION DENIED capturing for {source_ip}. "
              f"Raw sockets need root/sudo (outside Docker) or "
              f"cap_add: [NET_RAW, NET_ADMIN] (inside Docker). Skipping capture, "
              f"honeypot continues normally.")
        _active_captures.pop(source_ip, None)
    except Exception as e:
        print(f"[packet_capture] error capturing for {source_ip}: {e}")
        _active_captures.pop(source_ip, None)


def start_capture_for_ip(source_ip: str):
    """
    Non-blocking: starts a background thread that captures traffic to/from
    source_ip for CAPTURE_DURATION_SECONDS, then writes a .pcap file and
    exits. Safe to call repeatedly -- if a capture is already running for
    this IP, this is a no-op rather than stacking captures.
    """
    if source_ip in _active_captures:
        return
    thread = threading.Thread(target=_run_capture, args=(source_ip,), daemon=True)
    thread.start()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ml/packet_capture.py <ip_to_capture>")
        sys.exit(1)
    start_capture_for_ip(sys.argv[1])
    print(f"Capturing for {CAPTURE_DURATION_SECONDS}s, Ctrl+C to exit early (capture keeps running in background thread)...")
    time.sleep(CAPTURE_DURATION_SECONDS + 2)