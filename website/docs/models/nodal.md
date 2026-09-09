---
title: Nodal
description: Generators sited at buses, grouped into bus balance rows by a lookup parameter.
---

# Nodal

`nimopt.models.nodal` groups generators into buses through a lookup
parameter. `at[G, B]` has an entry where generator `g` sits at bus `b`, and
multiplying the generation by it maps a row over generators to a row over
buses. The coefficient introduces `B`, a dimension no variable has, so the
balance is indexed over the dimensions the lookup defines.

```text
minimise    Σ_{t,g} cost[g] · gen[t,g]
subject to  Σ_g at[g,b] · gen[t,g] == demand[b,t]    for each bus b and hour t
            0 ≤ gen[t,g] ≤ p_max[g]
```

```python
from nimopt.models import nodal

print(nodal.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
nodal  min  not built
  sets        T · G · B
  parameters  at (G,B) · p_max (G) · cost (G) · demand (B,T)
  variables   gen (T×G) [0.0, p_max]
  constraint  balance (B,T)  Sum(G, at[G, B] * gen[T, G]) == demand[B, T]
  objective   min  Sum(T, G, cost[G] * gen[T, G])
```

</details>
<!-- /output -->

Each bus meets its own demand from the generators sited at it, so the
optimum is a merit order per bus and hour.

```python
from nimopt.models import nodal

inputs = nodal.data()
solution = nodal.definition().build(inputs).solve()
print(solution.objective, nodal.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
11850.0 11850.0
```

</details>
<!-- /output -->

Every bus-hour has a row. A generator sited elsewhere is a term the lookup
removes from that row, not a row that is dropped.

```python
from nimopt.models import nodal

model = nodal.definition().build(nodal.data())
print(model.absent("balance"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
balance  6 of 6 rows  stated by terms
  term absent G='g0_0', B='b1_0', T=0  gen  absent-coefficient (at)
  term absent G='g0_0', B='b1_0', T=1  gen  absent-coefficient (at)
  term absent G='g0_0', B='b1_0', T=2  gen  absent-coefficient (at)
  term absent G='g1_0', B='b1_0', T=0  gen  absent-coefficient (at)
  term absent G='g1_0', B='b1_0', T=1  gen  absent-coefficient (at)
  term absent G='g1_0', B='b1_0', T=2  gen  absent-coefficient (at)
  term absent G='g2_0', B='b0_0', T=0  gen  absent-coefficient (at)
  term absent G='g2_0', B='b0_0', T=1  gen  absent-coefficient (at)
  term absent G='g2_0', B='b0_0', T=2  gen  absent-coefficient (at)
  term absent G='g3_0', B='b0_0', T=0  gen  absent-coefficient (at)
  term absent G='g3_0', B='b0_0', T=1  gen  absent-coefficient (at)
  term absent G='g3_0', B='b0_0', T=2  gen  absent-coefficient (at)
```

</details>
<!-- /output -->
