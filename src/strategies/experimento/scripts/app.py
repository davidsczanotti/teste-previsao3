from __future__ import annotations

import sqlite3
from pathlib import Path
from flask import Flask, render_template_string, request, send_from_directory, redirect, url_for
from pathlib import Path
from .report_wfo import build_wfo_artifacts_from_db, latest_wfo_group_from_db
from importlib import import_module


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
    # Latest WFO group indicator
    root = get_artifacts_root()
    latest_group = latest_wfo_group_from_db(str(get_db_path()))
    latest_dir = (root / latest_group) if latest_group else None
    has_files = latest_dir.exists() if latest_dir else False
    html = """
    <h1>Experimento — Runs</h1>
    <p>
      <a href="/wfo" style="padding:8px 12px;background:#2563eb;color:white;text-decoration:none;border-radius:6px;">Ir para WFO</a>
    </p>
    <div style=\"margin:12px 0;padding:10px;border:1px solid #ddd;border-radius:6px;\">
      <b>WFO mais recente:</b>
      {% if latest_group %}
        <code>{{ latest_group }}</code>
        {% if has_files %}
          — <a href=\"/artifacts/{{ latest_group }}/wfo_summary.json\">Resumo</a>
          — <a href=\"/artifacts/{{ latest_group }}/equity_curve.csv\">Equity CSV</a>
          — <a href=\"/wfo/{{ latest_group }}\">Ver janelas</a>
        {% else %}
          (sem arquivos ainda) 
          <form method=\"post\" action=\"/wfo/rebuild/latest\" style=\"display:inline-block; margin-left:8px;\">
            <button type=\"submit\" style=\"padding:4px 8px;background:#16a34a;color:white;border:none;border-radius:6px;\">Regenerar</button>
          </form>
        {% endif %}
      {% else %}
        (não há grupo WFO registrado no DB)
      {% endif %}
    </div>
    <form method="post" action="/pipeline/run" style="margin:10px 0;display:inline-block;">
      <button type="submit" style="padding:8px 12px;background:#0ea5e9;color:white;border:none;border-radius:6px;">Executar Pipeline (Backtest → MC → Report)</button>
    </form>
    <form method="post" action="/pipeline_wfo/run" style="margin:10px 0;display:inline-block;margin-left:8px;">
      <button type="submit" style="padding:8px 12px;background:#7c3aed;color:white;border:none;border-radius:6px;">Executar Pipeline WFO</button>
    </form>
    <form method="post" action="/cleanup/run" style="margin:10px 0;display:inline-block;margin-left:8px;">
      <button type="submit" style="padding:8px 12px;background:#ef4444;color:white;border:none;border-radius:6px;">Limpar artifacts (manter último WFO)</button>
    </form>
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
    return render_template_string(html, runs=runs, latest_group=latest_group, has_files=has_files)


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
    # Latest WFO group (from DB)
    latest_group = latest_wfo_group_from_db(str(get_db_path()))
    latest_dir = (root / latest_group) if latest_group else None
    has_files = latest_dir.exists() if latest_dir else False
    html = """
    <h1>WFO Artifacts</h1>
    <form method="post" action="/pipeline_wfo/run" style="margin:10px 0;">
      <button type="submit" style="padding:8px 12px;background:#7c3aed;color:white;border:none;border-radius:6px;">Executar Pipeline WFO (update → WFO → relatório)</button>
    </form>
    <form method="post" action="/wfo/rebuild/latest" style="margin:10px 0;">
      <button type="submit" style="padding:8px 12px;background:#16a34a;color:white;border:none;border-radius:6px;">Regenerar WFO mais recente (a partir do DB)</button>
    </form>
    <form method="post" action="/cleanup/run" style="margin:10px 0;">
      <button type="submit" style="padding:8px 12px;background:#ef4444;color:white;border:none;border-radius:6px;">Limpar artifacts (manter último WFO)</button>
    </form>
    <div style="margin:12px 0;padding:10px;border:1px solid #ddd;border-radius:6px;">
      <b>WFO mais recente:</b>
      {% if latest_group %}
        <code>{{ latest_group }}</code>
        {% if has_files %}
          — <a href="/artifacts/{{ latest_group }}/wfo_summary.json">Resumo</a>
          — <a href="/artifacts/{{ latest_group }}/equity_curve.csv">Equity CSV</a>
          — <a href="/artifacts/{{ latest_group }}/windows/">Windows</a>
        {% else %}
          (sem arquivos ainda — use "Regenerar WFO mais recente")
        {% endif %}
      {% else %}
        (não há grupo WFO registrado no DB)
      {% endif %}
    </div>
    <ul>
    {% for d in dirs %}
      <li><a href="/wfo/{{ d }}">{{ d }}</a></li>
    {% endfor %}
    </ul>
    """
    return render_template_string(html, dirs=dirs, latest_group=latest_group, has_files=has_files)


@app.route("/wfo/<name>")
def wfo_detail(name: str):
    root = get_artifacts_root()
    d = root / name
    if not d.exists():
        return "Not found", 404
    windows = sorted([p.name for p in (d / "windows").glob("window_*_candles.png")])
    html = """
    <h1>WFO {{ name }}</h1>
    <form method="post" action="/wfo/{{ name }}/rebuild" style="margin:10px 0;">
      <button type="submit" style="padding:8px 12px;background:#16a34a;color:white;border:none;border-radius:6px;">Regenerar este WFO (a partir do DB)</button>
      <a href="/wfo" style="margin-left:8px;padding:8px 12px;background:#2563eb;color:white;text-decoration:none;border-radius:6px;">Voltar WFO</a>
    </form>
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


@app.route("/wfo/rebuild/latest", methods=["POST"]) 
def wfo_rebuild_latest():
    group = latest_wfo_group_from_db(str(get_db_path()))
    if not group:
        return "No WFO group found in DB", 404
    out = get_artifacts_root() / group
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    build_wfo_artifacts_from_db(cfg, group, out)
    return redirect(url_for('wfo_detail', name=group))


@app.route("/wfo/<name>/rebuild", methods=["POST"]) 
def wfo_rebuild_name(name: str):
    out = get_artifacts_root() / name
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    build_wfo_artifacts_from_db(cfg, name, out)
    return redirect(url_for('wfo_detail', name=name))


@app.route("/cleanup/run", methods=["POST"]) 
def cleanup_run():
    try:
        mod = import_module("src.strategies.experimento.scripts.cleanup")
        mod.main()
        return redirect(url_for('wfo_index'))
    except Exception as e:
        return f"Cleanup error: {e}", 500


@app.route("/pipeline/run", methods=["POST"]) 
def run_pipeline():
    # pipeline: backtest -> monte_carlo -> report
    try:
        mod = import_module("src.strategies.experimento.scripts.pipeline")
        mod.main()
        return redirect(url_for('index'))
    except Exception as e:
        return f"Pipeline error: {e}", 500


@app.route("/pipeline_wfo/run", methods=["POST"]) 
def run_pipeline_wfo():
    try:
        mod = import_module("src.strategies.experimento.scripts.pipeline_wfo")
        mod.main()
        return redirect(url_for('wfo_index'))
    except Exception as e:
        return f"Pipeline WFO error: {e}", 500


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
