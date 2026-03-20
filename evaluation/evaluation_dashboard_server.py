#!/usr/bin/env python3
"""
Standalone Evaluation Dashboard Server.

Runs on a separate port and pulls data from a running simulation server.

Examples:
  python evaluation_dashboard_server.py
  python evaluation_dashboard_server.py --simulation-url http://127.0.0.1:8544 --port 8660
"""

from __future__ import annotations

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
from datetime import datetime
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def fetch_json(url: str, timeout: float = 3.0) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def build_html(simulation_url: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Evaluation Dashboard Proxy</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b1220; color:#e5e7eb; margin:0; padding:16px; }}
    .top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:12px; flex-wrap:wrap; }}
    .card-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; }}
    .card {{ background:#111827; border:1px solid #374151; border-radius:8px; padding:10px; }}
    .title {{ font-size:12px; font-weight:700; margin-bottom:6px; }}
    .v {{ font-size:13px; margin:3px 0; }}
    .ok {{ color:#86efac; }}
    .warn {{ color:#fca5a5; }}
    .meta {{ color:#9ca3af; font-size:12px; }}
    .series {{ margin-top:10px; background:#111827; border:1px solid #374151; border-radius:8px; padding:10px; }}
    table {{ width:100%; border-collapse: collapse; font-size:12px; }}
    th, td {{ border-bottom:1px solid #1f2937; padding:6px; text-align:left; }}
    th {{ color:#93c5fd; }}
    code {{ color:#93c5fd; }}
    button {{ background:#2563eb; color:white; border:none; border-radius:6px; padding:8px 12px; cursor:pointer; }}
  </style>
</head>
<body>
  <div class=\"top\">
    <div>
      <h2 style=\"margin:0;\">📊 Evaluation Dashboard (Separate Server)</h2>
      <div class=\"meta\">Pull source: <code>{simulation_url}</code></div>
    </div>
    <div>
      <button onclick=\"refreshNow()\">Refresh</button>
    </div>
  </div>

  <div id=\"status\" class=\"meta\">Loading...</div>

  <div class=\"card-grid\" style=\"margin-top:10px;\">
    <div class=\"card\">
      <div class=\"title\" style=\"color:#93c5fd;\">Mission Efficacy</div>
      <div class=\"v\">FHC: <b id=\"fhc\">-</b></div>
      <div class=\"v\">Area coverage rate: <b id=\"acr\">-</b></div>
      <div class=\"v\">Survivor detected rate: <b id=\"sdr\">-</b></div>
    </div>
    <div class=\"card\">
      <div class=\"title\" style=\"color:#86efac;\">Swarm Intelligence</div>
      <div class=\"v\">Exploration score: <b id=\"es\">-</b></div>
      <div class=\"v\">Redundancy penalty: <b id=\"rp\">-</b></div>
      <div class=\"v\">Dispersion penalty: <b id=\"dp\">-</b></div>
      <div class=\"v\">Average battery: <b id=\"ab\">-</b></div>
    </div>
    <div class=\"card\">
      <div class=\"title\" style=\"color:#fca5a5;\">Agentic + MCP</div>
      <div class=\"v\">Tool call accuracy: <b id=\"tca\">-</b></div>
      <div class=\"v\">Discovery latency: <b id=\"dl\">-</b></div>
      <div class=\"v\">Calls valid/total: <b id=\"calls\">-</b></div>
    </div>
  </div>

  <div class=\"series\">
    <div class=\"title\">Recent Time-Series (last 20 points)</div>
    <table>
      <thead>
        <tr>
          <th>Tick</th><th>Coverage%</th><th>SurvivorRate%</th><th>Exploration</th><th>ToolAccuracy%</th><th>LatencyTicks</th><th>WaitCount</th>
        </tr>
      </thead>
      <tbody id=\"seriesBody\"></tbody>
    </table>
  </div>

  <script>
    async function getJson(path) {{
      const r = await fetch(path);
      return await r.json();
    }}

    function fmt(v, d=2) {{
      const n = Number(v);
      if (Number.isNaN(n)) return '-';
      return n.toFixed(d);
    }}

    function setText(id, text) {{
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }}

    async function refreshNow() {{
      try {{
        const snapResp = await getJson('/api/snapshot');
        const seriesResp = await getJson('/api/series?limit=20');

        if (!snapResp.ok) throw new Error(snapResp.error || 'snapshot failed');
        const s = snapResp.snapshot || {{}};

        setText('status', `Connected. Scenario=${{s.scenario || 'n/a'}} split=${{String(s.split || 'n/a').toUpperCase()}} tick=${{s.tick ?? '-'}} (updated ${{new Date().toLocaleTimeString()}})`);
        setText('fhc', Number(s.fhc_ticks) >= 0 ? `${{s.fhc_ticks}} ticks / ${{fmt(s.fhc_seconds,2)}}s` : 'pending');
        setText('acr', fmt(s.area_coverage_rate, 3));
        setText('sdr', `${{fmt(s.survivor_detected_rate, 1)}}%`);

        setText('es', fmt(s.exploration_score, 1));
        setText('rp', fmt(s.redundancy_penalty, 3));
        setText('dp', fmt(s.dispersion_penalty, 3));
        setText('ab', fmt(s.avg_battery, 1));
        setText('dwc', String(s.drone_wait_count ?? 0));

        setText('tca', `${{fmt(s.tool_call_accuracy, 1)}}%`);
        setText('dl', `${{fmt(s.discovery_latency_ticks, 2)}} ticks / ${{fmt(s.discovery_latency_seconds, 2)}}s`);
        setText('calls', `${{s.tool_valid ?? 0}}/${{s.tool_total ?? 0}}`);

        const series = (seriesResp && seriesResp.ok && Array.isArray(seriesResp.series)) ? seriesResp.series : [];
        const body = document.getElementById('seriesBody');
        body.innerHTML = '';
        for (const row of series.slice().reverse()) {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${{row.tick ?? '-'}}<\/td>
            <td>${{fmt(row.coverage_pct, 2)}}<\/td>
            <td>${{fmt(row.survivor_detected_rate, 1)}}<\/td>
            <td>${{fmt(row.exploration_score, 1)}}<\/td>
            <td>${{fmt(row.tool_call_accuracy, 1)}}<\/td>
            <td>${{fmt(row.discovery_latency_ticks, 2)}}<\/td>
            <td>${{row.drone_wait_count ?? 0}}<\/td>
          `;
          body.appendChild(tr);
        }}
      }} catch (e) {{
        setText('status', 'Disconnected from simulation source: ' + String(e));
      }}
    }}

    refreshNow();
    setInterval(refreshNow, 2000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    simulation_url: str = "http://127.0.0.1:8544"

    def _write_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _write_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._write_html(build_html(self.simulation_url))
            return

        if path == "/api/snapshot":
            try:
                payload = fetch_json(f"{self.simulation_url.rstrip('/')}/metrics_snapshot")
                self._write_json(payload)
                return
            except (URLError, HTTPError, TimeoutError, ValueError) as exc:
                self._write_json({"ok": False, "error": str(exc)}, status=502)
                return

        if path == "/api/series":
            try:
                q = parsed.query or ""
                limit = 120
                for frag in q.split("&"):
                    if frag.startswith("limit="):
                        limit = int(frag.split("=", 1)[1])
                payload = fetch_json(f"{self.simulation_url.rstrip('/')}/metrics_series?limit={max(1,min(2000,limit))}")
                self._write_json(payload)
                return
            except (URLError, HTTPError, TimeoutError, ValueError) as exc:
                self._write_json({"ok": False, "error": str(exc)}, status=502)
                return

        self._write_json({"ok": False, "error": "not found"}, status=404)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone evaluation dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8660)
    parser.add_argument("--simulation-url", default="http://127.0.0.1:8544")
    args = parser.parse_args()

    handler_cls = type("ConfiguredDashboardHandler", (DashboardHandler,), {})
    handler_cls.simulation_url = str(args.simulation_url)

    server = ThreadingHTTPServer((args.host, int(args.port)), handler_cls)
    print(f"[{datetime.now().isoformat()}] Dashboard server running at http://{args.host}:{args.port}")
    print(f"Pulling simulation metrics from: {args.simulation_url}")
    print("Required simulation endpoints: /metrics_snapshot and /metrics_series")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
