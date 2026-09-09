"""Worked models, each stating its math, its inputs and its own answer.

A model here is one module exposing `definition()`, `data(scale=1)` and
`reference(data)`. The definition binds no data, so the documentation reads it
with `explain()`; `data(scale)` states inputs at any size, so a benchmark
builds one; and `reference` computes the objective by arithmetic over those
inputs, so a test has an answer that did not come from a solver.

Each module is imported by name — `from nimopt.models import dispatch` — and
this package imports none of them, so naming one costs one.
"""

MODELS = (
    "commitment",
    "dispatch",
    "expansion",
    "fleet",
    "nodal",
    "profiled",
    "recourse",
    "sector",
    "storage",
    "transport",
)

__all__ = ["MODELS"]
