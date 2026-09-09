---
title: Expansion
description: "A two-stage stochastic program: capacity chosen before the scenario is known, dispatched once it is."
---

# Expansion

`nimopt.models.expansion` is a two-stage stochastic program. The first stage
builds capacity; the second stage dispatches it against a demand and a fuel
price that the scenario reveals. What the first stage cannot see is stated by
the shape of its variable: `cap` is indexed by technology alone, so one
capacity serves every scenario, while `p` and `shed` carry the scenario and
may differ across it.

```text
minimise    Σ_g capital[g] · cap[g]
            + Σ_{s,g,t} weight[s] · cost[s,g] · p[s,g,t]
            + Σ_{s,t}   weight[s] · voll[s]  · shed[s,t]
subject to  p[s,g,t] ≤ cap[g]                              for each s, g, t
            Σ_g p[s,g,t] + shed[s,t] == demand[s,t]        for each s, t
            cap, p, shed ≥ 0
```

`weight` is the probability of a scenario, so the second and third sums are
an expectation and the objective is the capital committed plus the expected
cost of the recourse. There is no row tying one scenario's capacity to
another's: non-anticipativity is the missing dimension.

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

`cap` has one column per technology and `p` has one per scenario, technology
and hour. The capacity row is stated over all three dimensions even though
the variable it bounds carries one, so a single column is read by every
scenario's rows — which is what makes the capacity a shared decision.

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

A technology is available in full wherever it is built and nothing couples
one hour to the next, so the recourse in each scenario-hour is the merit
order of the built capacity against that demand, with the remainder unserved
at `voll`. Write the capacity as bands — the cheapest technology's band, then
the next, and last the band between the dearest technology and lost load —
and the total separates into one term per band, each convex in that band's
level and turning only at a demand. `reference` minimises them one at a time
and adds them up, so the answer is arithmetic over the inputs rather than a
second solve.

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

The data states a cost frontier: capital falls as marginal cost rises, and
lost load is dearer than the dearest technology. All three are built, and the
capacity that stacks to 180 leaves the cold scenario's peak hour of 225
short — shedding 45 is cheaper than a fourth band that earns its capital in
one hour of one scenario.

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

The separation holds only for data whose technologies are a frontier and
whose bands stack. `reference` refuses anything else rather than returning a
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
ValueError: capital does not fall from base to what follows it
```

</details>
<!-- /output -->
