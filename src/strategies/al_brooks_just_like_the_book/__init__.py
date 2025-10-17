"""Al Brooks (book-style) strategy package.

This module encodes a programmable approximation of Al Brooks' price action
methodology, with clear, auditable heuristics for the most common setups:
trend continuation via inside bars (ii/ioi), H2/L2 in-trend entries, and
breakout + pullback (BO-PB) entries. It integrates with the project's
optimization and walk-forward validation utilities.
"""

