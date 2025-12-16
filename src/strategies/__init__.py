"""Strategy namespace package (compatibility layer).

This repo historically referenced `src.strategies.<name>.*`. The concrete
implementations currently live under `src.core.*`, but we keep this package so
existing CLIs/tests/imports keep working.
"""

