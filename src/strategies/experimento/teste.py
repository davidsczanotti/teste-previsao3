python - << 'PY'
from pathlib import Path

p = Path("src/strategies/experimento/scripts/optimize.py")
s = p.read_text(encoding="utf-8")

old = 'df["atr_30m"] = atr(df, length=int(params.get("atr_len", cfg["indicators"][2]["params"]["length"])) )'

new = (
' # Dynamic ATR length from config or fallback 14\n'
' atr_default = 14\n'
' for ind in cfg.get("indicators", []):\n'
' if ind.get("name") == "atr" and ind.get("tf") == cfg["base_timeframe"]:\n'
' atr_default = int(ind.get("params", {}).get("length", atr_default))\n'
' break\n'
' df["atr_30m"] = atr(df, length=int(params.get("atr_len", atr_default)) )'
)

if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print("OK")
else:
    print("NOT FOUND")
PY