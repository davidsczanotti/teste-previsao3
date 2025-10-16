"""Simple EMA-only strategy (mean reversion by default).

Provides a tiny, didactic baseline to compare against more complex
approaches. Uses cached klines in `data/klines_cache.db` via the
existing loader utilities when requested.
"""

