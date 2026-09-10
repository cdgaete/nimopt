---
title: Fleet
description: The dispatch model declared as one variable per unit, instead of one variable over a product.
---

# Fleet

`nimopt.models.fleet` is the same problem as `dispatch`. It is declared as
one variable per unit over the snapshots alone, and the terms of the units
are added into one balance row. The optimum is the same merit order, and the
cost of the declaration differs.

```text
minimise    Σ_t Σ_u cost_u[t] · u[t]
subject to  Σ_u u[t] == load[t]          for each snapshot t
            0 ≤ u[t] ≤ p_max_u[t]        for each unit u
```

`definition` takes a scale here: the number of variables is a property of
the declaration, not of the data.

```python
from nimopt.models import fleet

print(fleet.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
fleet  min  not built
  sets        T
  parameters  load (T) · p_max_g0_0 (T) · cost_g0_0 (T) · p_max_g1_0 (T) · cost_g1_0 (T) · p_max_g2_0 (T) · cost_g2_0 (T)
  variables   g0_0 (T) [0.0, p_max_g0_0] · g1_0 (T) [0.0, p_max_g1_0] · g2_0 (T) [0.0, p_max_g2_0]
  constraint  balance (T)  g0_0[T] + g1_0[T] + g2_0[T] == load[T]
  objective   min  Sum(T, cost_g0_0[T] * g0_0[T]) + Sum(T, cost_g1_0[T] * g1_0[T]) + Sum(T, cost_g2_0[T] * g2_0[T])
```

</details>
<!-- /output -->

```python
from nimopt.models import fleet

inputs = fleet.data()
model = fleet.definition().build(inputs)
solution = model.solve()
print(len(model.variables), "variables,", model.n_columns, "columns")
print(solution.objective, fleet.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3 variables, 12 columns
8900.0 8900.0
```

</details>
<!-- /output -->

One balance row per hour, and every unit appears in every row.

```python
from nimopt.models import fleet

model = fleet.definition().build(fleet.data())
print(model.absent("balance"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
balance  4 of 4 rows  stated by terms
```

</details>
<!-- /output -->
