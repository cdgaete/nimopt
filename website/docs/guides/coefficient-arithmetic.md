---
title: Coefficient arithmetic
description: Combine parameters into a coefficient, divide by a parameter, carry a constant, and the forms that are refused with the form to write instead.
---

# Coefficient arithmetic

A coefficient is often derived from several parameters: fuel price divided
by efficiency, a cost scaled by a factor. In `nimopt` a coefficient is a
parameter read at its sets or an arithmetic combination of such readings.
`+`, `-`, `*`, `/` and a power by a number combine them. The combination is
symbolic: it holds references, derives its dimensions from its operands, and
is evaluated once, when the term it multiplies is materialised. A derived
coefficient can therefore appear in a definition before any data exists.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["base", "peak"]))
T = Set("T", np.arange(3))
price = Param.from_dense("fuel_price", (G, T), np.full((2, 3), 30.0))
eta = Param.from_dense("efficiency", (G, T), np.array([[0.5] * 3, [0.4] * 3]))
load = Param.from_dense("load", (T,), np.full(3, 100.0))
cap = Param.from_dense("capacity", (G, T), np.full((2, 3), 80.0))

unit_cost = price[G, T] / eta[G, T]

m = Model("dispatch", sense="min")
gen = m.var("gen", (G, T), lower=0.0, upper=cap)
m.eq("balance", Sum(G, gen[G, T]) == load[T])
m.set_objective(Sum(G, T, unit_cost[G, T] * gen[G, T]))

print(unit_cost.name, unit_cost.dims)
print(m.solve().objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
(fuel_price / efficiency) ('G', 'T')
18900.0
```

</details>
<!-- /output -->

## Reading a derived coefficient

`unit_cost[G, T]` reads a combination the same way `price[G, T]` reads a
parameter, and the sets given are checked against the combination's
dimensions. A transposed or incomplete index raises `ValueError`.

```python raises=ValueError
import numpy as np
from nimopt import Param, Set

G = Set("G", np.array(["base", "peak"]))
T = Set("T", np.arange(3))
price = Param.from_dense("fuel_price", (G, T), np.full((2, 3), 30.0))
eta = Param.from_dense("efficiency", (G, T), np.array([[0.5] * 3, [0.4] * 3]))

(price[G, T] / eta[G, T])[T, G]
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: coefficient (fuel_price / efficiency) is over ('G', 'T'); got ('T', 'G')
```

</details>
<!-- /output -->

A bare parameter has no arithmetic: `price * 2.0` raises `TypeError`. Read
the parameter at its sets first and combine the references.

## Alignment

Two operands with the same dimensions align entry by entry. Operands whose
dimensions nest or overlap align on the shared dimensions and broadcast over
the rest, with the left operand's order first. Operands sharing no dimension
raise `ValueError`: their product would be an outer product, which a linear
model does not require. The same rule applies when a coefficient multiplies
a variable. A number has no dimensions and scales.

```python raises=ValueError
import numpy as np
from nimopt import Param, Set

G = Set("G", np.array(["base", "peak"]))
T = Set("T", np.arange(3))
over_g = Param.from_dense("over_g", (G,), np.ones(2))
over_t = Param.from_dense("over_t", (T,), np.ones(3))

over_g[G] * over_t[T]
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: frames ('G',) and ('T',) share no dimension; there is nothing to align them on
```

</details>
<!-- /output -->

**A coefficient never adds a column the variable does not have.** The
variable's members define the columns; a coefficient can only reduce which
of them receive a nonzero. A coefficient with dimensions the variable lacks
defines rows over those dimensions: this is how a term maps rows to columns.

## Division by zero

A divisor that is zero raises `ZeroDivisionError`; the message gives the
coordinate. Substituting infinity would hand the solver a model nobody
wrote. The check covers a Python number, a NumPy scalar and a coefficient
with a zero at any coordinate, because a NumPy scalar divides to infinity
where a Python number raises.

```python raises=ZeroDivisionError
import numpy as np
from nimopt import Param, Set

G = Set("G", np.array(["base", "peak"]))
T = Set("T", np.arange(3))
price = Param.from_dense("fuel_price", (G, T), np.full((2, 3), 30.0))
holed = Param.from_dense("holed", (G, T), np.array([[0.5, 0.0, 0.5], [0.4] * 3]))

price[G, T] / holed[G, T]
```

<!-- output -->
<details open>
<summary>Raises ZeroDivisionError</summary>

```text
ZeroDivisionError: divisor holed carries a zero at 1 coordinate(s), the first at {'G': 'base', 'T': 1}; a quotient there states a coefficient no solver can read
```

</details>
<!-- /output -->

## Constants

An expression consists of terms and a constant. `x + 1 <= 5` produces the
row `x <= 4`: the constant moves to the right-hand side when the constraint
is built, and into the reported objective value as a fixed cost. A constant
adds no column. A constant alone is neither an objective nor a constraint,
and raises.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["a"]))
one = Param.from_dense("one", (P,), np.ones(1))

m = Model("m", sense="max")
x = m.var("x", (P,), upper=100.0)
m.eq("cap", one[P] * x[P] + 1.0 <= 5.0)
m.set_objective(Sum(P, one[P] * x[P]) + 7.0)

print(m.assemble().n_cols, m.assemble().row_upper)
print(m.solve().objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
1 [4.]
11.0
```

</details>
<!-- /output -->

## Refused forms

A line of modelling arithmetic produces a linear term, or raises with a
message that gives the form to write instead. There is no third outcome.

| Written | The message says to write |
| --- | --- |
| `x[P] * y[P]` | a linear term has one variable; a coefficient multiplies it |
| `x[P] ** 2` | a coefficient takes the power, and a variable multiplies it |
| `x[P] / y[P]` | a variable in a denominator is not linear |
| `2.0 / x[P]` | the same: write the reciprocal as a coefficient |
| `x[P] / 0.0` | a divisor of zero is handled before it reaches an expression |
| `x[P] < 1.0` | `<=` and `>=`; an LP has no row for a strict inequality |
| `x[P] > 1.0` | the same |
| `x[P] != 1.0` | one bound per equation |
| `0.0 <= x[P] <= 1.0` | each bound as its own equation |
| `x[P] + c[P]` | a coefficient has no row until a variable multiplies it |
| `abs(x[P])`, `min` and `max` | reduce with `Sum`, or bound the expression with two rows |
| `np.sum(x[P])` | `Sum` and its sets |
| `np.array([...]) * x[P]` | `Param.from_dense`, read at its sets |
| `Sum(P, Sum(P, x[P]))` | a dimension is reduced once |
| `Sum(P - 1, x[P])` | `Sum(P, x[P - 1])`: the lag belongs on the reference |
| `T - 1.7` | a lag is a whole number of members |

Each raises a `nimopt` message rather than a bare Python error, and the test
suite executes both the table and the rewrites the messages name.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["a", "b"]))
m = Model("m")
x = m.var("x", (P,))

np.array([1.0, 2.0]) * x[P]
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: a coefficient is a parameter; build one with `Param.from_dense` or `Param.from_long` and read it at its sets. A product of two expressions is not linear.
```

</details>
<!-- /output -->

A NumPy array multiplying a term would let NumPy handle the operator and
return an array of expressions. The expression types refuse the ufunc, and
the message says how a coefficient is built.
