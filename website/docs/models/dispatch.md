---
title: Dispatch
description: Least-cost dispatch of a generator fleet against a load. The baseline formulation.
---

# Dispatch

`nimopt.models.dispatch` is least-cost dispatch of a generator fleet against
a load. One variable `p` is indexed over snapshots and generators, there is
one balance row per snapshot, and each generator has a cost. Every other
model in the corpus adds one axis to this one.

```text
minimize    Σ_{t,g} cost[g] · p[t,g]
subject to  Σ_g p[t,g] == load[t]        for each snapshot t
            0 ≤ p[t,g] ≤ p_max[g]
```

The balance row has no coefficient. A sum over a dimension requires none,
and the corpus writes no coefficient a model does not require.

```python
from nimopt.models import dispatch

print(dispatch.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
dispatch  min  not built
  sets        snapshot · generator
  parameters  p_max (generator) · load (snapshot) · cost (generator)
  variables   p (snapshot×generator) [0.0, p_max]
  constraint  balance (snapshot)  Sum(generator, p[snapshot, generator]) == load[snapshot]
  objective   min  Sum(snapshot, generator, cost[generator] * p[snapshot, generator])
```

</details>
<!-- /output -->

Snapshots are independent, and the optimum is the merit order per snapshot.
`reference` computes it without a solver.

```python
from nimopt.models import dispatch

inputs = dispatch.data()
solution = dispatch.definition().build(inputs).solve()
print(solution.status)
print(solution.objective, dispatch.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal
1920.0 1920.0
```

</details>
<!-- /output -->

The balance row is produced for every snapshot in the load, and `absent`
reports no dropped row.

```python
from nimopt.models import dispatch

model = dispatch.definition().build(dispatch.data())
print(model.absent("balance"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
balance  6 of 6 rows  stated by terms
```

</details>
<!-- /output -->
