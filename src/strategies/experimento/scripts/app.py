from __future__ import annotations

import sqlite3
from pathlib import Path
from flask import Flask, render_template_string, request, send_from_directory
from pathlib import Path


app = Flask(__name__)


def get_db_path() -> str:
    # Read config lazily to avoid coupling
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg["storage"]["results_db"]


def get_artifacts_root() -> Path:
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return Path(cfg["storage"]["artifacts_dir"]).resolve()


@app.route("/")
def index():
    with sqlite3.connect(get_db_path()) as cx:
        runs = cx.execute("SELECT run_id, started_at, finished_at FROM runs ORDER BY started_at DESC LIMIT 200").fetchall()
    html = """
    <h1>Experimento — Runs</h1>
    <table border=1 cellpadding=4>
      <tr><th>run_id</th><th>started_at</th><th>finished_at</th></tr>
      {% for r in runs %}
      <tr>
        <td><a href="/run/{{ r[0] }}">{{ r[0] }}</a></td>
        <td>{{ r[1] }}</td>
        <td>{{ r[2] }}</td>
      </tr>
      {% endfor %}
    </table>
    """
    return render_template_string(html, runs=runs)


@app.route("/run/<run_id>")
def run_detail(run_id: str):
    with sqlite3.connect(get_db_path()) as cx:
        bars = cx.execute("SELECT idx, close_time, close, signal FROM bars WHERE run_id=? ORDER BY idx", (run_id,)).fetchall()
        trades = cx.execute("SELECT trade_id, entry_time, exit_time, side, qty, entry_price, exit_price, pnl FROM trades WHERE run_id=? ORDER BY trade_id", (run_id,)).fetchall()
        metrics = cx.execute("SELECT key, value FROM metrics WHERE run_id=?", (run_id,)).fetchall()
    html = """
    <h1>Run {{ run_id }}</h1>
    <h2>Metrics</h2>
    <ul>
    {% for k,v in metrics %}
      <li>{{ k }}: {{ '%.6f'|format(v) }}</li>
    {% endfor %}
    </ul>
    <h2>Trades</h2>
    <table border=1 cellpadding=4>
      <tr><th>id</th><th>entry_time</th><th>exit_time</th><th>side</th><th>qty</th><th>entry</th><th>exit</th><th>pnl</th></tr>
      {% for t in trades %}
      <tr>
        <td>{{ t[0] }}</td><td>{{ t[1] }}</td><td>{{ t[2] }}</td><td>{{ t[3] }}</td>
        <td>{{ '%.6f'|format(t[4]) }}</td><td>{{ '%.2f'|format(t[5]) }}</td><td>{{ '%.2f'|format(t[6]) }}</td><td>{{ '%.2f'|format(t[7]) }}</td>
      </tr>
      {% endfor %}
    </table>
    <h2>Bars (sample)</h2>
    <table border=1 cellpadding=4>
      <tr><th>idx</th><th>time</th><th>close</th><th>signal</th></tr>
      {% for b in bars[:300] %}
      <tr>
        <td>{{ b[0] }}</td><td>{{ b[1] }}</td><td>{{ '%.2f'|format(b[2]) }}</td><td>{{ b[3] }}</td>
      </tr>
      {% endfor %}
    </table>
    """
    return render_template_string(html, run_id=run_id, bars=bars, trades=trades, metrics=metrics)


@app.route("/wfo")
def wfo_index():
    root = get_artifacts_root()
    dirs = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("wfo-")])
    html = """
    <h1>WFO Artifacts</h1>
    <ul>
    {% for d in dirs %}
      <li><a href="/wfo/{{ d }}">{{ d }}</a></li>
    {% endfor %}
    </ul>
    """
    return render_template_string(html, dirs=dirs)


@app.route("/wfo/<name>")
def wfo_detail(name: str):
    root = get_artifacts_root()
    d = root / name
    if not d.exists():
        return "Not found", 404
    windows = sorted([p.name for p in (d / "windows").glob("window_*_candles.png")])
    html = """
    <h1>WFO {{ name }}</h1>
    <p>
      <a href="/artifacts/{{ name }}/wfo_summary.json">wfo_summary.json</a> |
      <a href="/artifacts/{{ name }}/equity_curve.csv">equity_curve.csv</a> |
      <a href="/artifacts/{{ name }}/wfo_equity.png">wfo_equity.png</a> |
      <a href="/artifacts/{{ name }}/wfo_windows.png">wfo_windows.png</a>
    </p>
    <h2>Windows</h2>
    <ul>
    {% for w in windows %}
      {% set base = w.replace('_candles.png','') %}
      <li>
        <a href="/artifacts/{{ name }}/windows/{{ w }}">{{ w }}</a>
        — <a href="/artifacts/{{ name }}/windows/{{ base }}_equity.png">equity</a>
        — <a href="/artifacts/{{ name }}/windows/{{ base }}_equity.csv">equity.csv</a>
        — <a href="/artifacts/{{ name }}/windows/{{ base }}_params.json">params</a>
      </li>
    {% endfor %}
    </ul>
    """
    return render_template_string(html, name=name, windows=windows)


@app.route("/artifacts/<path:subpath>")
def artifacts(subpath: str):
    root = get_artifacts_root()
    # Ensure path traversal protection
    full = (root / subpath).resolve()
    if root not in full.parents and full != root:
        return "Invalid path", 400
    if not full.exists():
        return "Not found", 404
    if full.is_dir():
        entries = sorted([p.name for p in full.iterdir()])
        html = """
        <h1>Artifacts: {{ path }}</h1>
        <ul>
        {% for e in entries %}
          <li><a href="/artifacts/{{ path }}/{{ e }}">{{ e }}</a></li>
        {% endfor %}
        </ul>
        """
        return render_template_string(html, path=subpath, entries=entries)
    return send_from_directory(full.parent, full.name)


def main():
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()
