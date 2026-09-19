"""Worked models, each with its formulation, its inputs and its own answer.

A model here is one module with `definition()`, `data(scale=1)` and
`reference(data)`. The definition binds no data, and `explain()` reads it.
`data(scale)` returns inputs at any size. `reference` computes the objective
by arithmetic over those inputs, and a test compares it against a solved
model. Each module is imported by name, and this package imports none.

`fleet.definition(scale=1)` also takes the scale argument. It declares one
variable per unit. The scale then sets the count of declared variables, not
only the size of the data. `transport.definition(integer=False)` and
`transport.data(scale=1, integer=False)` take an integer argument. It selects
the bounded integer form of the flow variable. `data` then adds the capacity
parameter that bounds it.
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
