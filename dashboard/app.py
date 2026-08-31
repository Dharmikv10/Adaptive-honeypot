"""
dashboard/app.py

Phase 2 dashboard: intentionally basic. Shows that the pipeline (services ->
SQLite -> web view) works end to end, before Phase 5 turns this into the
full SOC-style interface (WebSockets, animations, ML pipeline visualization,
attack replay). Two JSON endpoints + one HTML page that polls them.

Run standalone:
    python3 dashboard/app.py
Then visit http://<vm-ip>:9000 in a browser (or http://127.0.0.1:9000
if you're browsing from inside the VM, e.g. via a forwarded port).
"""

import asyncio
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import DB_PATH  # single source of truth -- see db/database.py

LISTEN_PORT = 9000

app = FastAPI()


def _query(sql: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/logs")
async def api_logs():
    rows = await asyncio.to_thread(
        _query,
        """SELECT timestamp, source_ip, protocol, dest_port, event_type,
                  username, password, payload
           FROM attack_logs ORDER BY id DESC LIMIT 30""",
    )
    return JSONResponse(rows)


@app.get("/api/sessions")
async def api_sessions():
    """
    Phase 4 addition: surfaces the ML classification + adaptation decision
    for each session, joined with the IP's current policy state. This is
    what was missing from the dashboard before -- /api/logs only ever showed
    raw attack_logs rows, never the predicted_class/confidence/reason/
    adaptation_taken columns that ml/adaption.py writes via
    record_session_intelligence().
    """
    rows = await asyncio.to_thread(
        _query,
        """SELECT s.session_id, s.source_ip, s.protocol, s.start_time,
                  s.event_count, s.command_count,
                  s.predicted_class, s.confidence, s.reason, s.adaptation_taken,
                  s.threat_score, p.tarpit, p.escalated_logging, p.decoy_expanded
           FROM sessions s
           LEFT JOIN ip_policy p ON p.source_ip = s.source_ip
           WHERE s.predicted_class IS NOT NULL
           ORDER BY s.start_time DESC LIMIT 30""",
    )
    return JSONResponse(rows)


@app.get("/api/stats")
async def api_stats():
    def _stats():
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM attack_logs").fetchone()[0]
        unique_ips = conn.execute(
            "SELECT COUNT(DISTINCT source_ip) FROM attack_logs"
        ).fetchone()[0]
        by_protocol = dict(
            conn.execute(
                "SELECT protocol, COUNT(*) FROM attack_logs GROUP BY protocol"
            ).fetchall()
        )
        conn.close()
        return {"total": total, "unique_ips": unique_ips, "by_protocol": by_protocol}

    return JSONResponse(await asyncio.to_thread(_stats))


PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Adaptive Honeypot -- Command Center</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0a0e14; --panel: #11161f; --panel-2: #161d29; --border: #232b39;
    --text: #c9d3e0; --muted: #6b7789; --accent: #4fd1ff; --accent-2: #7c5cff;
    --green: #3ddc97; --amber: #ffb454; --red: #ff5d7a;
    --mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text);
    margin: 0; padding: 2rem clamp(1rem, 4vw, 3rem);
    background-image:
      radial-gradient(circle at 15% 0%, rgba(79,209,255,0.08), transparent 40%),
      radial-gradient(circle at 85% 20%, rgba(124,92,255,0.08), transparent 40%);
  }
  header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.75rem; flex-wrap: wrap; gap: 0.75rem; }
  .brand { display: flex; align-items: center; gap: 0.75rem; }
  .brand-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    box-shadow: 0 0 24px rgba(79,209,255,0.35);
  }
  h1 { font-size: 1.35rem; margin: 0; color: #f0f4f8; letter-spacing: 0.2px; }
  .subtitle { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
  .live-badge {
    display: flex; align-items: center; gap: 6px; background: var(--panel);
    border: 1px solid var(--border); padding: 6px 12px; border-radius: 999px;
    font-size: 0.75rem; color: var(--green);
  }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.75rem; }
  .stat-card {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.2rem; position: relative; overflow: hidden;
  }
  .stat-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
  }
  .stat-card .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .stat-card .value { font-size: 1.9rem; font-weight: 700; color: #f0f4f8; margin-top: 0.3rem; font-family: var(--mono); }
  .stat-card .value.threat-low { color: var(--green); }
  .stat-card .value.threat-mid { color: var(--amber); }
  .stat-card .value.threat-high { color: var(--red); }

  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; margin-bottom: 1.75rem; }
  .panel h2 { font-size: 0.95rem; color: #dfe6ee; margin: 0 0 0.9rem 0; display: flex; align-items: center; gap: 8px; }
  .panel h2::before { content: ""; width: 4px; height: 16px; border-radius: 2px; background: linear-gradient(180deg, var(--accent), var(--accent-2)); display: inline-block; }

  .proto-bars { display: flex; flex-direction: column; gap: 8px; margin-bottom: 0.5rem; }
  .proto-bar-row { display: grid; grid-template-columns: 70px 1fr 34px; align-items: center; gap: 10px; font-size: 0.78rem; }
  .proto-bar-track { background: #0d1219; border-radius: 6px; height: 9px; overflow: hidden; border: 1px solid var(--border); }
  .proto-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }

  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
  td { color: #d5dde8; }
  td.mono, .mono { font-family: var(--mono); }
  tbody tr { transition: background 0.15s ease; }
  tbody tr:hover { background: rgba(79,209,255,0.05); }
  .table-scroll { overflow-x: auto; }

  .proto { padding: 3px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; display: inline-block; }
  .SSH { background: rgba(79,209,255,0.12); color: var(--accent); border: 1px solid rgba(79,209,255,0.3); }
  .HTTP { background: rgba(61,220,151,0.12); color: var(--green); border: 1px solid rgba(61,220,151,0.3); }
  .FTP { background: rgba(124,92,255,0.12); color: #a996ff; border: 1px solid rgba(124,92,255,0.3); }
  .TELNET { background: rgba(255,180,84,0.12); color: var(--amber); border: 1px solid rgba(255,180,84,0.3); }
  .MYSQL { background: rgba(255,93,122,0.12); color: var(--red); border: 1px solid rgba(255,93,122,0.3); }

  .cred { color: #d5dde8; }
  .cred.empty { color: var(--muted); }
  .payload-cell { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }

  .conf-wrap { display: flex; align-items: center; gap: 8px; min-width: 110px; }
  .conf-track { flex: 1; height: 6px; background: #0d1219; border-radius: 4px; overflow: hidden; border: 1px solid var(--border); }
  .conf-fill { height: 100%; border-radius: 4px; }
  .conf-label { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); min-width: 32px; }

  .action-tag { display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 999px; font-size: 0.68rem; background: rgba(255,93,122,0.12); color: var(--red); border: 1px solid rgba(255,93,122,0.3); }
  .action-tag.none { background: rgba(107,119,137,0.12); color: var(--muted); border: 1px solid var(--border); }

  .threat-pill { padding: 3px 9px; border-radius: 999px; font-size: 0.72rem; font-family: var(--mono); font-weight: 600; }
  .empty-state { color: var(--muted); font-size: 0.82rem; padding: 1.2rem 0; text-align: center; }
</style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-icon">🛡️</div>
      <div>
        <h1>Adaptive Honeypot -- Command Center</h1>
        <div class="subtitle">Real-time deception telemetry &amp; ML-driven threat response</div>
      </div>
    </div>
    <div class="live-badge"><span class="live-dot"></span> LIVE</div>
  </header>

  <div class="stats-grid" id="stats"></div>

  <div class="panel">
    <h2>Protocol Distribution</h2>
    <div class="proto-bars" id="protoBars"></div>
  </div>

  <div class="panel">
    <h2>Live Attack Feed</h2>
    <div class="table-scroll">
      <table>
        <thead>
          <tr><th>Time</th><th>Source IP</th><th>Protocol</th><th>Port</th><th>Event</th><th>Username</th><th>Password</th><th>Payload</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>ML Predictions &amp; Adaptation</h2>
    <div class="table-scroll">
      <table>
        <thead>
          <tr><th>Time</th><th>Source IP</th><th>Protocol</th><th>Predicted Class</th><th>Confidence</th><th>Reason</th><th>Actions Taken</th><th>Threat Score</th></tr>
        </thead>
        <tbody id="sessionRows"></tbody>
      </table>
    </div>
  </div>

<script>
const PROTO_COLORS = { SSH: '#4fd1ff', HTTP: '#3ddc97', FTP: '#a996ff', TELNET: '#ffb454', MYSQL: '#ff5d7a' };

function threatClass(score) {
  if (score >= 7) return { cls: 'threat-high', color: 'var(--red)', bg: 'rgba(255,93,122,0.15)' };
  if (score >= 4) return { cls: 'threat-mid', color: 'var(--amber)', bg: 'rgba(255,180,84,0.15)' };
  return { cls: 'threat-low', color: 'var(--green)', bg: 'rgba(61,220,151,0.15)' };
}

function timeOnly(iso) {
  if (!iso) return '';
  const parts = iso.split('T');
  return parts[1] ? parts[1].split('.')[0] : iso;
}

function escapeHtml(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function refresh() {
  const [stats, logs, sessions] = await Promise.all([
    fetch('/api/stats').then(r => r.json()),
    fetch('/api/logs').then(r => r.json()),
    fetch('/api/sessions').then(r => r.json()),
  ]);

  const maxThreat = sessions.reduce((m, s) => Math.max(m, s.threat_score || 0), 0);
  const t = threatClass(maxThreat);

  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="label">Total Events</div><div class="value">${stats.total}</div></div>
    <div class="stat-card"><div class="label">Unique Attackers</div><div class="value">${stats.unique_ips}</div></div>
    <div class="stat-card"><div class="label">Protocols Active</div><div class="value">${Object.keys(stats.by_protocol).length}</div></div>
    <div class="stat-card"><div class="label">Peak Threat Score</div><div class="value ${t.cls}">${maxThreat.toFixed(1)}<span style="font-size:0.9rem;color:var(--muted);">/10</span></div></div>
  `;

  const protoTotal = Object.values(stats.by_protocol).reduce((a, b) => a + b, 0) || 1;
  const protoEntries = Object.entries(stats.by_protocol);
  document.getElementById('protoBars').innerHTML = protoEntries.length ? protoEntries.map(([proto, count]) => {
    const pct = Math.round((count / protoTotal) * 100);
    const color = PROTO_COLORS[proto] || '#8b949e';
    return `
    <div class="proto-bar-row">
      <span class="proto ${proto}">${proto}</span>
      <div class="proto-bar-track"><div class="proto-bar-fill" style="width:${pct}%; background:${color};"></div></div>
      <span class="mono" style="color:var(--muted)">${count}</span>
    </div>`;
  }).join('') : '<div class="empty-state">No traffic recorded yet</div>';

  document.getElementById('rows').innerHTML = logs.length ? logs.map(r => `
    <tr>
      <td class="mono" style="color:var(--muted)">${timeOnly(r.timestamp)}</td>
      <td class="mono">${r.source_ip}</td>
      <td><span class="proto ${r.protocol}">${r.protocol}</span></td>
      <td class="mono" style="color:var(--muted)">${r.dest_port ?? ''}</td>
      <td>${r.event_type ?? ''}</td>
      <td class="mono cred ${r.username ? '' : 'empty'}">${r.username ? escapeHtml(r.username) : '--'}</td>
      <td class="mono cred ${r.password ? '' : 'empty'}">${r.password ? escapeHtml(r.password) : '--'}</td>
      <td class="payload-cell" title="${escapeHtml(r.payload ?? '')}">${escapeHtml((r.payload ?? '--').toString().slice(0, 60))}</td>
    </tr>
  `).join('') : '<tr><td colspan="8"><div class="empty-state">No events yet -- waiting for the first connection...</div></td></tr>';

  document.getElementById('sessionRows').innerHTML = sessions.length ? sessions.map(s => {
    const actions = (s.adaptation_taken || '')
      .split(',').map(a => a.trim()).filter(Boolean)
      .map(a => `<span class="action-tag">${a}</span>`).join('');
    const conf = s.confidence != null ? s.confidence : 0;
    const confColor = conf >= 0.75 ? 'var(--red)' : conf >= 0.5 ? 'var(--amber)' : 'var(--green)';
    const st = threatClass(s.threat_score ?? 0);
    return `
    <tr>
      <td class="mono" style="color:var(--muted)">${timeOnly(s.start_time)}</td>
      <td class="mono">${s.source_ip}</td>
      <td><span class="proto ${s.protocol}">${s.protocol}</span></td>
      <td>${s.predicted_class ?? '--'}</td>
      <td>
        <div class="conf-wrap">
          <div class="conf-track"><div class="conf-fill" style="width:${(conf*100).toFixed(0)}%; background:${confColor};"></div></div>
          <span class="conf-label">${s.confidence != null ? (conf * 100).toFixed(0) + '%' : '--'}</span>
        </div>
      </td>
      <td style="white-space:normal; max-width:280px; color:var(--muted); font-size:0.78rem;">${s.reason ?? ''}</td>
      <td>${actions || '<span class="action-tag none">none</span>'}</td>
      <td><span class="threat-pill" style="color:${st.color}; background:${st.bg};">${(s.threat_score ?? 0).toFixed(1)}/10</span></td>
    </tr>
  `;
  }).join('') : '<tr><td colspan="8"><div class="empty-state">No classified sessions yet</div></td></tr>';
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


async def serve():
    config = uvicorn.Config(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
    server = uvicorn.Server(config)
    print(f"[dashboard] Listening on port {LISTEN_PORT} ...")
    await server.serve()


def run():
    asyncio.run(serve())


if __name__ == "__main__":
    run()