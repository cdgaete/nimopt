---
title: Storage
description: A generator fleet and batteries meeting an hourly load, with the hours coupled through the stored energy.
---

# Storage

`nimopt.models.storage` dispatches a generator fleet and a set of batteries
against an hourly load. The `state_of_charge` row references the previous
hour through `T.cyclic - 1`. The row at the first hour therefore references
the last hour, and every hour has a row. The ramp row references `T - 1`. The
first hour has no predecessor, and that row is not produced. The model
exercises both lag rules.

```text
minimize    Σ_{g,t} cost[g,t] · gen[g,t]
subject to  Σ_g gen[g,t] + Σ_s discharge[s,t] − Σ_s charge[s,t] == load[t]
            soc[s,t] − soc[s,t−1] − charge_eta[s,t] · charge[s,t]
                + discharge_eta[s,t] · discharge[s,t] == 0     (t−1 wraps)
            gen[g,t] − gen[g,t−1] ≤ ramp_limit[g,t]              (t=0 dropped)
            gen[g,t] ≤ capacity[g,t]
            charge[s,t] ≤ power[s,t]
            discharge[s,t] ≤ power[s,t]
            soc[s,t] ≤ energy[s,t]
```

Every limit is a constraint and not a bound, and each therefore has a dual
value.

```python
from nimopt.models import storage

print(storage.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
storage  min  not built
  sets        T · G · S
  parameters  cost (G,T) · capacity (G,T) · ramp_limit (G,T) · load (T) · power (S,T) · energy (S,T) · charge_eta (S,T) · discharge_eta (S,T)
  variables   gen (G×T) [0.0, inf] · charge (S×T) [0.0, inf] · discharge (S×T) [0.0, inf] · soc (S×T) [0.0, inf]
  constraint  balance (T)  Sum(G, gen[G, T]) + Sum(S, discharge[S, T]) - Sum(S, charge[S, T]) == load[T]
  constraint  state_of_charge (S,T)  soc[S, T] - soc[S, T.cyclic - 1] - charge_eta[S, T] * charge[S, T] + discharge_eta[S, T] * discharge[S, T] == 0
  constraint  generation_limit (G,T)  gen[G, T] <= capacity[G, T]
  constraint  ramp (G,T)  gen[G, T] - gen[G, T - 1] <= ramp_limit[G, T]
  constraint  charge_limit (S,T)  charge[S, T] <= power[S, T]
  constraint  discharge_limit (S,T)  discharge[S, T] <= power[S, T]
  constraint  energy_limit (S,T)  soc[S, T] <= energy[S, T]
  objective   min  Sum(G, T, cost[G, T] * gen[G, T])
```

</details>
<!-- /output -->

The batteries are lossy, with a round-trip efficiency of `0.95 · 0.93`, and
the costs of the fleet span 50.0 to 55.0. Shifting energy through the store
costs more than it saves. The store is therefore idle, and the optimum is the
hourly merit order. A store that cycles requires data with a wider cost
spread, and the benchmarks supply it.

```python
from nimopt.models import storage

inputs = storage.data()
solution = storage.definition().build(inputs).solve()
print(solution.objective, storage.reference(inputs))
print("store moved:", abs(solution.primal("charge").to_dense()).max())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
145449.3739227024 145449.37392270242
store moved: 0.0
```

</details>
<!-- /output -->

Every hour has a `state_of_charge` row: that lag wraps. The ramp row at the
first hour is not produced: that lag does not wrap.

```python
from nimopt.models import storage

model = storage.definition().build(storage.data())
print(model.absent("state_of_charge"))
print()
print(model.absent("ramp"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
state_of_charge  24 of 24 rows  stated by terms

ramp  69 of 72 rows  stated by terms
  row absent  G='base0', T=0  term-does-not-reach (gen)
  row absent  G='mid0', T=0  term-does-not-reach (gen)
  row absent  G='peak0', T=0  term-does-not-reach (gen)
```

</details>
<!-- /output -->
