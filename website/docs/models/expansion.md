---
title: Expansion
description: "A two-stage stochastic program: capacity chosen before the scenario is known, dispatched once it is."
---

# Expansion

`nimopt.models.expansion` is a two-stage stochastic program. The first stage
builds capacity. The second stage dispatches it against a demand and a fuel
price given by the scenario. The shape of the first-stage variable expresses
the information available to it: `cap` is indexed by technology alone, and one
capacity applies to every scenario. `p` and `shed` are indexed by scenario as
well and may differ across it.

```text
minimise    Σ_g capital[g] · cap[g]
            + Σ_{s,g,t} weight[s] · cost[s,g] · p[s,g,t]
            + Σ_{s,t}   weight[s] · voll[s]  · shed[s,t]
subject to  p[s,g,t] ≤ cap[g]                              for each s, g, t
            Σ_g p[s,g,t] + shed[s,t] == demand[s,t]        for each s, t
            cap, p, shed ≥ 0
```

`weight` is the probability of a scenario. The second and third sums are
therefore an expectation, and the objective is the committed capital plus the
expected cost of the recourse. No row relates the capacity of one scenario to
the capacity of another: the missing dimension expresses
non-anticipativity.

```python
from nimopt.models import expansion

print(expansion.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
expansion  min  not built
  sets        S · G · T
  parameters  capital (G) · cost (S,G) · demand (S,T) · weight (S) · voll (S)
  variables   cap (G) [0.0, inf] · p (S×G×T) [0.0, inf] · shed (S×T) [0.0, inf]
  constraint  capacity (S,G,T)  p[S, G, T] - cap[G] <= 0
  constraint  balance (S,T)  Sum(G, p[S, G, T]) + shed[S, T] == demand[S, T]
  objective   min  Sum(G, capital[G] * cap[G]) + Sum(S, G, T, (weight[S] * cost[S, G]) * p[S, G, T]) + Sum(S, T, (weight[S] * voll[S]) * shed[S, T])
```

</details>
<!-- /output -->

`cap` has one column per technology, and `p` has one per scenario,
technology and hour. The capacity row is declared over all three dimensions,
and the variable it bounds has one. The rows of every scenario read the same
column, and the capacity is therefore a shared decision.

```python
from nimopt.models import expansion

model = expansion.definition().build(expansion.data())
print(model.explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
expansion  min  75 columns · 72 rows · 180 nonzeros
  sets        G 3 · S 3 · T 6
  parameters  demand (S,T) 18 · capital (G) 3 · weight (S) 3 · cost (S,G) 9 · voll (S) 3
  variables   cap (G) 3 cols [0.0, inf] · p (S×G×T) 54 cols [0.0, inf] · shed (S×T) 18 cols [0.0, inf]
  constraint  capacity (S,G,T)  p[S, G, T] - cap[G] <= 0  54 rows  108 nz
  constraint  balance (S,T)  Sum(G, p[S, G, T]) + shed[S, T] == demand[S, T]  18 rows  72 nz
  objective   min  Sum(G, capital[G] * cap[G]) + Sum(S, G, T, (weight[S] * cost[S, G]) * p[S, G, T]) + Sum(S, T, (weight[S] * voll[S]) * shed[S, T])
```

</details>
<!-- /output -->

A technology is available in full wherever it is built, and no row couples
one hour to the next. The recourse in each scenario-hour is therefore the
merit order of the built capacity against that demand, with the remainder
unserved at `voll`. The capacity is written as bands: the band of the
cheapest technology, then the next, and last the band between the most
expensive technology and lost load. The total separates into one term per
band. Each term is convex in the level of its band and changes slope only at
a demand. `reference` minimizes the terms one at a time and adds them, and
the result is arithmetic over the inputs, not a second solve.

```python
from nimopt.models import expansion

inputs = expansion.data()
solution = expansion.definition().build(inputs).solve()
print(solution.objective, expansion.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
29703.899999999994 29703.899999999994
```

</details>
<!-- /output -->

The data describes a cost frontier: capital falls as marginal cost rises,
and lost load is more expensive than the most expensive technology. All three
technologies are built. The capacity stacks to 180 and leaves the peak hour
of 225 in the cold scenario short. Shedding 45 costs less than a fourth band
used in one hour of one scenario.

```python
import numpy as np

from nimopt.models import expansion

inputs = expansion.data()
solution = expansion.definition().build(inputs).solve()
print(np.cumsum(np.asarray(solution.primal("cap").values())))
print(np.asarray(solution.primal("shed").values()).reshape(3, 6))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[119. 153. 180.]
[[ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0. 45.  0.  0.]]
```

</details>
<!-- /output -->

The separation applies only to data whose technologies form a frontier and
whose bands stack. `reference` raises on any other data, and it returns no
number that is not the optimum.

```python raises=ValueError
from nimopt.models import expansion

inputs = expansion.data()
inputs["capital"] = inputs["capital"][::-1]
print(expansion.reference(inputs))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: capital does not fall from base to what follows it; pass a capital cost that falls across the merit order
```

</details>
<!-- /output -->
