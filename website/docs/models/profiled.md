---
title: Profiled
description: A dispatch whose capacity is a profile, with one bound per generator and hour.
---

# Profiled

`nimopt.models.profiled` is a dispatch whose capacity varies by hour.
`dispatch` bounds a generator by a single number. Here `p_max` is a
parameter over generators and snapshots. A solar unit is bounded by its
hourly availability, and a thermal unit by its rating.

```text
minimize    Σ_{t,g} cost[g] · gen[t,g]
subject to  Σ_g gen[t,g] == load[t]          for each snapshot t
            0 ≤ gen[t,g] ≤ profile[g,t]
```

The profile is indexed `(G, T)` and the variable `(T, G)`. A bound is read
in the dimension order of the variable it bounds, and both orderings select
the same columns.

```python
from nimopt.models import profiled

print(profiled.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
profiled  min  not built
  sets        T · G
  parameters  profile (G,T) · cost (G) · load (T)
  variables   gen (T×G) [0.0, profile]
  constraint  balance (T)  Sum(G, gen[T, G]) == load[T]
  objective   min  Sum(T, G, cost[G] * gen[T, G])
```

</details>
<!-- /output -->

Each snapshot is independent. The optimum is the merit order against the
capacities of that hour.

```python
from nimopt.models import profiled

inputs = profiled.data()
solution = profiled.definition().build(inputs).solve()
print(solution.objective, profiled.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
28119.536003699297 28119.536003699293
```

</details>
<!-- /output -->

Every hour has a balance row. A generator whose profile is zero is a column
bounded to zero, not a dropped row.

```python
from nimopt.models import profiled

model = profiled.definition().build(profiled.data())
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
