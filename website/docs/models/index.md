---
title: Worked models
description: Ten complete models, each with its formulation, its data at any scale, and an independently computed reference objective.
---

# Worked models

`nimopt.models` contains ten models. Each is a module with three functions.

| Name | Returns |
| --- | --- |
| `definition()` | a `Definition`: the formulation with no data bound |
| `data(scale=1)` | the inputs, at a given size |
| `reference(data)` | the optimal objective, computed by direct arithmetic |

The three serve different readers. This documentation calls `explain()`, a
benchmark calls `build(data(100))`, and a test compares a solve against
`reference(data(1))`. A reference is arithmetic over the inputs that uses
nothing from `nimopt`, so a formulation error is not checked against a copy
of itself.

| Model | Exercises |
| --- | --- |
| [`dispatch`](/models/dispatch) | the baseline formulation |
| [`transport`](/models/transport) | a sparse network over a subset of a product |
| [`storage`](/models/storage) | temporal coupling and a cyclic state |
| [`nodal`](/models/nodal) | grouping through a lookup parameter |
| [`commitment`](/models/commitment) | binary columns |
| [`fleet`](/models/fleet) | many small declarations rather than one large one |
| [`profiled`](/models/profiled) | a bound that varies by hour |
| [`sector`](/models/sector) | mixed density: dense in one axis, sparse in another |
| [`expansion`](/models/expansion) | a two-stage stochastic program: capacity before the scenario, dispatch after |
| [`recourse`](/models/recourse) | a binary first stage taken before the scenario is known |

```python
from nimopt.models import dispatch

inputs = dispatch.data()
solution = dispatch.definition().build(inputs).solve()
print(solution.status, solution.objective, dispatch.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal 1920.0 1920.0
```

</details>
<!-- /output -->
