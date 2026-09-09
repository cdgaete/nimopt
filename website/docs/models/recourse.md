---
title: Recourse
description: "Stochastic unit commitment: a binary on-off decision fixed before the scenario is known."
---

# Recourse

`nimopt.models.recourse` is `commitment` under uncertainty. The demand and the
fuel price are a scenario, and the on-off decision is taken before either is
revealed: `on` is indexed by hour and unit alone, while `p` and `shed` carry
the scenario. One commitment has to serve every scenario, which is what makes
the binary a hedge rather than a schedule.

```text
minimise    Σ_{t,g}   no_load[g] · on[t,g]
            + Σ_{s,g,t} weight[s] · cost[s,g] · p[s,g,t]
            + Σ_{s,t}   weight[s] · voll[s]  · shed[s,t]
subject to  p[s,g,t] ≤ p_max[g] · on[t,g]                  for each s, g, t
            p[s,g,t] ≥ p_min[g] · on[t,g]                  for each s, g, t
            Σ_g p[s,g,t] + shed[s,t] == demand[s,t]        for each s, t
            on ∈ {0,1},  p, shed ≥ 0
```

The first stage costs what it costs whatever happens; the second and third
sums are weighted by the probability of a scenario, so the objective is a
commitment charge plus the expected cost of the recourse.

```python
from nimopt.models import recourse

print(recourse.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
recourse  min  not built
  sets        S · G · T
  parameters  p_max (G) · p_min (G) · no_load (G) · cost (S,G) · demand (S,T) · weight (S) · voll (S)
  variables   on (T×G) [0.0, 1.0] integer · p (S×G×T) [0.0, inf] · shed (S×T) [0.0, inf]
  constraint  capacity (S,G,T)  p[S, G, T] - p_max[G] * on[T, G] <= 0
  constraint  minimum (S,G,T)  p[S, G, T] - p_min[G] * on[T, G] >= 0
  constraint  balance (S,T)  Sum(G, p[S, G, T]) + shed[S, T] == demand[S, T]
  objective   min  Sum(T, G, no_load[G] * on[T, G]) + Sum(S, G, T, (weight[S] * cost[S, G]) * p[S, G, T]) + Sum(S, T, (weight[S] * voll[S]) * shed[S, T])
```

</details>
<!-- /output -->

Nothing couples one hour to the next, so the commitment is chosen hour by
hour. `reference` enumerates every on-off subset of the fleet and scores each
by its expected recourse across the scenarios, which is exact and cheap:
three units make eight subsets.

```python
from nimopt.models import recourse

inputs = recourse.data()
solution = recourse.definition().build(inputs).solve()
print(solution.objective, recourse.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
16744.8125 16744.8125
```

</details>
<!-- /output -->

A committed unit runs at least its minimum and there is nowhere to put
unwanted energy, so a unit whose minimum exceeds the mildest scenario's
demand cannot be committed at all — the row it would break is that scenario's
balance. The first hour asks 38.25 in the mildest scenario, and `base`, whose
minimum is 40.0 and whose fuel is the cheapest of the three, is left off it.
The third hour goes the other way: the fleet carries 260.0 against a coldest
demand of 268.75, so everything is committed and the remainder is shed.

```python
import numpy as np

from nimopt.models import recourse

inputs = recourse.data()
solution = recourse.definition().build(inputs).solve()
print(inputs["demand"].round(2))
print(np.asarray(solution.primal("on").values()).reshape(4, 3))
print(np.asarray(solution.primal("shed").values()).reshape(3, 4))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[[ 38.25 119.   182.75  80.75]
 [ 45.   140.   215.    95.  ]
 [ 56.25 175.   268.75 118.75]]
[[0. 1. 0.]
 [1. 1. 0.]
 [1. 1. 1.]
 [1. 0. 0.]]
[[0.   0.   0.   0.  ]
 [0.   0.   0.   0.  ]
 [0.   0.   8.75 0.  ]]
```

</details>
<!-- /output -->

The mean demand in the first hour is 43.875, which `base` can serve and is
cheapest at. A model given that one number commits it, and that commitment
breaks the balance of the mildest scenario, whose probability is 0.5. The
scenario dimension on `demand` is what rules the commitment out; the missing
scenario dimension on `on` is what makes the ruling out bind.
