---
title: Transport
description: Plants shipping to warehouses over a network that is not complete.
---

# Transport

`nimopt.models.transport` ships from plants to warehouses over an incomplete
network: a plant serves a band of nearby warehouses and not all of them. The
cost parameter has one entry per arc, and the flow variable takes its members
from that parameter. The model therefore has one column per arc, not one per
cell of the plant-warehouse product.

```text
minimise    Σ_{(p,w) ∈ arcs} cost[p,w] · flow[p,w]
subject to  Σ_w flow[p,w] ≤ supply[p]     for each plant p
            Σ_p flow[p,w] ≥ demand[w]     for each warehouse w
            flow[p,w] ≥ 0                 for each arc (p,w)
```

```python
from nimopt.models import transport

print(transport.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
transport  min  not built
  sets        P · W
  parameters  cost (P,W) · supply (P) · demand (W)
  variables   flow (P×W) over cost [0.0, inf]
  constraint  supply (P)  Sum(W, flow[P, W]) <= supply[P]
  constraint  demand (W)  Sum(P, flow[P, W]) >= demand[W]
  objective   min  Sum(P, W, cost[P, W] * flow[P, W])
```

</details>
<!-- /output -->

Supply is twice the total demand of the band of a plant. No supply row binds
therefore, and each warehouse buys from the cheapest plant connected to it.
`reference` computes that sum.

```python
from nimopt.models import transport

inputs = transport.data()
model = transport.definition().build(inputs)
solution = model.solve()
print(model.n_columns, "columns for", len(inputs["cost"][1]), "arcs")
print(solution.objective, transport.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
12 columns for 12 arcs
100.99601811246072 100.99601811246073
```

</details>
<!-- /output -->

Arcs are drawn from every warehouse but the last, so the last warehouse is
reached by no plant and has no demand row. `absent` reports the row and the
rule that dropped it.

```python
from nimopt.models import transport

model = transport.definition().build(transport.data())
print(model.absent("demand"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
demand  5 of 6 rows  stated by terms
  row absent  W='w5'  term-does-not-reach (flow)
```

</details>
<!-- /output -->
