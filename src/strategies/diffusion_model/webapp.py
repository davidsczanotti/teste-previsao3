from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template


@dataclass
class LiveState:
    data: Dict[str, Any] = field(default_factory=dict)
    history: list[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.data) if self.data else {}

    def set(self, payload: Dict[str, Any], meta: Dict[str, Any] | None = None) -> None:
        with self._lock:
            self.data = payload
            if meta is not None:
                self.history.append(meta)
                if len(self.history) > self.max_history:
                    self.history.pop(0)


def create_app(state: LiveState) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/live")
    def api_live():
        payload = state.get()
        if not payload:
            return jsonify({"status": "warming_up"})
        return jsonify(payload)

    @app.get("/api/snapshots")
    def api_snapshots():
        with state._lock:
            return jsonify(state.history[-state.max_history :])

    @app.get("/healthz")
    def healthz():
        return "ok"

    return app


def _run(app: Flask, host: str, port: int):
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def start_server(state: LiveState, host: str = "127.0.0.1", port: int = 5001, open_browser: bool = True) -> None:
    app = create_app(state)
    thread = threading.Thread(target=_run, args=(app, host, port), daemon=True)
    thread.start()
    if open_browser:
        # Give the server a moment to start
        time.sleep(0.8)
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass
