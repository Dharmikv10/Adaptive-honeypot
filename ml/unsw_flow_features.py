"""Experimental PCAP-to-flow adapter for the saved UNSW-NB15 models.

This is not the official UNSW-NB15 extractor. Features requiring dataset
specific processing or connection history are explicitly reported as
unavailable and set to zero as placeholders.
"""

from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import joblib
from scapy.all import IP, IPv6, TCP, UDP, ICMP, PcapReader

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "unsw_binary_xgboost_clean.joblib"
)

HISTORY_FEATURES = {
    "ct_dst_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_src_dport_ltm", "ct_src_ltm", "ct_srv_dst", "ct_srv_src",
    "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd",
    "is_ftp_login",
}
UNAVAILABLE_BY_DEFAULT = {"response_body_len", "trans_depth"}


def _ip_pair(packet):
    if IP in packet:
        ip = packet[IP]
        return ip.src, ip.dst, int(ip.ttl), int(ip.len)
    if IPv6 in packet:
        ip = packet[IPv6]
        return ip.src, ip.dst, int(ip.hlim), len(packet)
    return None


def _transport(packet):
    if TCP in packet:
        layer = packet[TCP]
        return "tcp", int(layer.sport), int(layer.dport)
    if UDP in packet:
        layer = packet[UDP]
        return "udp", int(layer.sport), int(layer.dport)
    if ICMP in packet:
        return "icmp", 0, 0
    return "other", 0, 0


def _service(port):
    return {
        20: "ftp-data", 21: "ftp", 22: "ssh", 25: "smtp",
        53: "dns", 67: "dhcp", 80: "http", 110: "pop3",
        443: "ssl", 161: "snmp", 1812: "radius", 6667: "irc",
        2223: "ssh", 8080: "http", 2121: "ftp",
        2323: "telnet", 3307: "mysql",
    }.get(port, "-")


def _flow_key(src, dst, sport, dport, proto):
    endpoints = tuple(sorted(((src, sport), (dst, dport))))
    return proto, endpoints


def extract_pcap_features(pcap_path):
    """Return (feature_rows, quality_report) from a PCAP file."""
    pcap_path = Path(pcap_path)
    if not pcap_path.is_file():
        raise FileNotFoundError(pcap_path)

    artifact = joblib.load(MODEL_PATH)
    columns = list(artifact["feature_columns"])
    flows = {}

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            ip_info = _ip_pair(packet)
            if ip_info is None:
                continue

            src, dst, ttl, ip_length = ip_info
            proto, sport, dport = _transport(packet)
            key = _flow_key(src, dst, sport, dport, proto)

            if key not in flows:
                flows[key] = {
                    "packets": [],
                    "start": float(packet.time),
                    "end": float(packet.time),
                    "src": src,
                    "dst": dst,
                    "sport": sport,
                    "dport": dport,
                    "proto": proto,
                    "ttl": [],
                    "payload_lengths": [],
                    "tcp_flags": [],
                    "synack": None,
                    "ackdat": None,
                }

            flow = flows[key]
            timestamp = float(packet.time)
            flow["start"] = min(flow["start"], timestamp)
            flow["end"] = max(flow["end"], timestamp)
            flow["packets"].append((timestamp, ip_length, src == flow["src"]))
            flow["ttl"].append(ttl)

            payload_length = len(bytes(packet.payload))
            flow["payload_lengths"].append(payload_length)

            if TCP in packet:
                tcp = packet[TCP]
                flags = int(tcp.flags)
                flow["tcp_flags"].append(flags)
                if flags & 0x12 == 0x12:
                    flow["synack"] = timestamp
                if flags & 0x10 and flow["ackdat"] is None:
                    flow["ackdat"] = timestamp

    rows = []
    unavailable_union = set(HISTORY_FEATURES) | set(UNAVAILABLE_BY_DEFAULT)

    for flow in flows.values():
        packets = sorted(flow["packets"], key=lambda item: item[0])
        times = [p[0] for p in packets]
        lengths = [p[1] for p in packets]
        directions = [p[2] for p in packets]
        duration = max(flow["end"] - flow["start"], 0.0)

        src_packets = [i for i, direction in enumerate(directions) if direction]
        dst_packets = [i for i, direction in enumerate(directions) if not direction]
        src_lengths = [lengths[i] for i in src_packets]
        dst_lengths = [lengths[i] for i in dst_packets]

        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        src_gaps = [
            times[b] - times[a]
            for a, b in zip(src_packets, src_packets[1:])
        ]
        dst_gaps = [
            times[b] - times[a]
            for a, b in zip(dst_packets, dst_packets[1:])
        ]

        def safe_mean(values):
            return float(mean(values)) if values else 0.0

        def safe_std(values):
            return float(pstdev(values)) if len(values) > 1 else 0.0

        def safe_rate(count, elapsed):
            return float(count / elapsed) if elapsed > 0 else 0.0

        features = {name: 0 for name in columns}
        features.update({
            "dur": duration,
            "spkts": len(src_packets),
            "dpkts": len(dst_packets),
            "sbytes": sum(src_lengths),
            "dbytes": sum(dst_lengths),
            "sttl": safe_mean([flow["ttl"][i] for i in range(len(directions))
                               if directions[i]]) if src_packets else 0,
            "dttl": safe_mean([flow["ttl"][i] for i in range(len(directions))
                               if not directions[i]]) if dst_packets else 0,
            "smean": safe_mean(src_lengths),
            "dmean": safe_mean(dst_lengths),
            "sinpkt": safe_mean(src_gaps),
            "dinpkt": safe_mean(dst_gaps),
            "sjit": safe_std(src_gaps),
            "djit": safe_std(dst_gaps),
            "rate": safe_rate(len(packets), duration),
            "sload": safe_rate(sum(src_lengths) * 8, duration),
            "dload": safe_rate(sum(dst_lengths) * 8, duration),
            "sloss": 0,
            "dloss": 0,
            "swin": 0,
            "dwin": 0,
            "synack": (
                max(flow["synack"] - flow["start"], 0.0)
                if flow["synack"] is not None else 0
            ),
            "ackdat": (
                max(flow["ackdat"] - flow["start"], 0.0)
                if flow["ackdat"] is not None else 0
            ),
            "tcprtt": 0,
            "ct_state_ttl": 0,
            "is_sm_ips_ports": int(
                flow["src"] == flow["dst"]
                and flow["sport"] == flow["dport"]
            ),
        })

        proto_name = flow["proto"]
        proto_column = f"proto_{proto_name}"
        if proto_column in features:
            features[proto_column] = 1

        service_column = f"service_{_service(flow['dport'])}"
        if service_column in features:
            features[service_column] = 1

        flags = flow["tcp_flags"]
        if flags:
            state = "CON"
            if any(f & 0x04 for f in flags):
                state = "RST"
            elif any(f & 0x01 for f in flags):
                state = "FIN"
            elif any(f & 0x02 for f in flags):
                state = "REQ"
            state_column = f"state_{state}"
            if state_column in features:
                features[state_column] = 1
        else:
            state_column = "state_INT"
            if state_column in features:
                features[state_column] = 1

        rows.append({
            "flow": {
                "src": flow["src"],
                "dst": flow["dst"],
                "src_port": flow["sport"],
                "dst_port": flow["dport"],
                "protocol": flow["proto"],
                "packet_count": len(packets),
                "duration_seconds": duration,
            },
            "features": {name: float(features[name]) for name in columns},
        })


    measured_features = {
        "dur", "spkts", "dpkts", "sbytes", "dbytes",
        "sttl", "dttl", "smean", "dmean",
        "sinpkt", "dinpkt", "sjit", "djit",
        "rate", "sload", "dload", "synack", "ackdat",
        "is_sm_ips_ports",
    }

    derived_features = {
        "proto", "service", "state",
    }

    placeholder_features = {
        "sloss", "dloss", "swin", "dwin", "tcprtt",
        "ct_state_ttl", "response_body_len", "trans_depth",
        "ct_dst_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "ct_flw_http_mthd", "ct_ftp_cmd", "ct_src_dport_ltm",
        "ct_src_ltm", "ct_srv_dst", "ct_srv_src", "is_ftp_login",
    }

    one_hot_features = {
        name for name in columns
        if name.startswith(("proto_", "service_", "state_"))
    }

    quality = {
        "pcap": str(pcap_path),
        "flow_count": len(rows),
        "model_feature_count": len(columns),
        "measured_features": sorted(
            name for name in measured_features if name in columns
        ),
        "measured_feature_count": sum(
            name in columns for name in measured_features
        ),
        "derived_feature_groups": sorted(derived_features),
        "one_hot_feature_count": len(one_hot_features),
        "placeholder_features": sorted(
            name for name in placeholder_features if name in columns
        ),
        "placeholder_feature_count": sum(
            name in columns for name in placeholder_features
        ),
        "unavailable_or_placeholder_features": sorted(
            name for name in placeholder_features if name in columns
        ),
        "warning": (
            "Experimental partial feature extraction. Some UNSW-NB15 "
            "features are placeholders, and several calculated features "
            "are approximations. Predictions may be unreliable on these "
            "partial network-flow features."
        ),
    }
    return rows, quality
