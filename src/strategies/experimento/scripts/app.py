from __future__ import annotations

import sqlite3
from pathlib import Path
from flask import Flask, render_template_string, request


app = Flask(__name__)


def get_db_path() -> str:
    # Read config lazily to avoid coupling
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg["storage"]["results_db"]


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


def main():
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()

