---
title: Sector
description: Technologies sited in some regions, running in every hour. Sparse in one axis, dense in the other.
---

# Sector

`nimopt.models.sector` has mixed density. The region-technology map is
sparse, since a technology exists in some regions and not others, while
every sited pair runs in every hour. The generation variable takes its
members from a parameter over the sited pairs crossed with the whole
horizon, so it is sparse in one axis and dense in the other.

```text
minimise    Σ_{(r,k) sited, t} cost[r,k] · gen[r,k,t]
subject to  Σ_k gen[r,k,t] == demand[r,t]     for each region r and hour t
            0 ≤ gen[r,k,t] ≤ capacity[r,k]    for each sited (r,k) and hour t
```

```python
from nimopt.models import sector

print(sector.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
sector  min  not built
  sets        R · K · T
  parameters  sited (R,K,T) · capacity (R,K) · cost (R,K) · demand (R,T)
  variables   gen (R×K×T) over sited [0.0, capacity]
  constraint  balance (R,T)  Sum(K, gen[R, K, T]) == demand[R, T]
  objective   min  Sum(R, K, T, cost[R, K] * gen[R, K, T])
```

</details>
<!-- /output -->

Each region meets its own demand from the technologies sited in it, so the
optimum is a merit order per region and hour.

```python
from nimopt.models import sector

inputs = sector.data()
model = sector.definition().build(inputs)
solution = model.solve()
print(
    model.n_columns,
    "columns of a possible",
    len(inputs["R"]) * len(inputs["K"]) * len(inputs["T"]),
)
print(solution.objective, sector.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
16 columns of a possible 24
18350.0 18350.0
```

</details>
<!-- /output -->

Every region-hour has a balance row, and the capacity bound applies only to
the sited pairs, so no balance row is dropped.

```python
from nimopt.models import sector

model = sector.definition().build(sector.data())
print(model.absent("balance"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
balance  8 of 8 rows  stated by terms
```

</details>
<!-- /output -->
