"""Worked models, each with its formulation, its inputs and its own answer.

A model here is one module with `definition()`, `data(scale=1)` and
`reference(data)`. The definition binds no data, and `explain()` reads it.
`data(scale)` returns inputs at any size. `reference` computes the objective
by arithmetic over those inputs, and a test compares it against a solved
model. Each module is imported by name, and this package imports none.
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
