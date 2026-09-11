---
title: Expressions
description: Terms, the frame they share, the sums that reduce them and the relations that bound them.
---

# Expressions

## `Term`

One variable, an optional coefficient, the dimensions summed over, and a
scale factor. A term describes a block of coefficients and contains
references, not arrays. Writing it allocates nothing: an expression over a
million columns costs the same as one over ten.

| Member | Returns |
| --- | --- |
| `free_dims` | the dimensions it is still indexed over |
| `carried_dims` | every dimension it has |
| `with_coefficient(coefficient)` | the term, scaled by a parameter |
| `summing(dims)` | the term, reduced over those dimensions |
| `scaled(by)` | the term, multiplied by a number |
| `restricted_to(domain)` | the term, over those members only |

A caller builds terms through the operators rather than these members:
`cost[P, W] * x[P, W]` gives a coefficient, `Sum` gives the reduction, and
`-` gives the scale.

## `Expression`

A list of terms and the frame they share. The frame is the union of the
terms' free dimensions, ordered by the term that introduces each. A term
narrower than the frame is broadcast over it when the expression is
materialised.

| Member | Returns |
| --- | --- |
| `terms` | the terms it contains |
| `frame` | the dimensions it is indexed over |
| `coords` | the coordinates of that frame |
| `materialise()` | its coefficients as a `nimblend` array |

```python
import numpy as np
from nimopt import Model, Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))

m = Model("transport")
x = m.var("x", (P, W))
y = m.var("y", (P, W))

combined = cost[P, W] * x[P, W] - y[P, W]
print(combined.frame)
print(len(combined.terms))
print(combined.terms[0].free_dims)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
2
('P', 'W')
```

</details>
<!-- /output -->

## `Sum`

```
Sum(I, J, ..., expression, where=None)
```

The expression reduced over the named sets. Each set named leaves the
frame. `where=` takes a domain and restricts the entries of each term before
the reduction. The sum then runs over the coordinates given, not over every
coordinate of the product.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

print(x[P, W].frame)
print(Sum(W, x[P, W]).frame)
print(Sum(P, W, x[P, W]).frame)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
('P',)
()
```

</details>
<!-- /output -->

A sum is over the members of a set and takes the set, not a lag of it. Write
the lag at the variable reference.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum

T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (T,))

Sum(T - 1, x[T])
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: a sum is over the members of ['T'] and takes the set, not a lag of it; write the lag at the variable's reference
```

</details>
<!-- /output -->

## `Relation`

An expression, a sense and a right-hand side, produced by comparing an
expression with `<=`, `>=` or `==`. `Model.constraint` turns one into a
constraint.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

bounded = Sum(W, x[P, W]) <= 30.0
print(type(bounded).__name__, bounded.sense)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
Relation <=
```

</details>
<!-- /output -->

A relation has no truth value. Python evaluates `0 <= expr <= 10` as two
comparisons joined by `and` and keeps only the second. The chained form
raises, and the first bound is not dropped.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

0.0 <= Sum(W, x[P, W]) <= 10.0
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: a relation has no truth value; write each bound in its own constraint
```

</details>
<!-- /output -->

## Forms that are not linear

`nimopt` expresses linear terms. Each form below raises where it is
written, and the message gives the form to write instead.

A variable raised to a power is not linear; a coefficient takes the power
and a variable multiplies it.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
x[T] ** 2
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: cannot raise an expression to a power: expressions are linear; raise a coefficient to the power and multiply it by a variable
```

</details>
<!-- /output -->

A variable in a denominator is not linear either; the reciprocal is written
as a coefficient the variable multiplies.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
1.0 / x[T]
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: cannot divide by an expression: expressions are linear; declare the reciprocal as a coefficient the variable multiplies
```

</details>
<!-- /output -->

The absolute value of an expression is not linear. A magnitude is written
with two rows bounding the expression, and a reduction with `Sum` over its
sets.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
abs(x[T])
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: an expression has no absolute value: expressions are linear; bound the expression with two rows, or reduce it with `Sum` over its sets
```

</details>
<!-- /output -->

An LP has no row for a strict inequality.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
x[T] < 5.0
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: an LP has no row for a strict inequality; write `<=` or `>=`, and reduce with `Sum` in place of `min` or `max`
```

</details>
<!-- /output -->

The built-in `sum` of expressions with no set to reduce over calls
`Expression.sum`. That call raises, and it returns no unreduced expression.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
x[T].sum()
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: an expression is reduced over the sets it is summed across; specify them with `Sum(I, J, expression)`
```

</details>
<!-- /output -->

A relation expresses one bound. Comparing it a second time raises, and the
first bound is not dropped.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
(x[T] <= 5.0) >= 1.0
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: a relation already has one bound; compare the expression again in its own constraint
```

</details>
<!-- /output -->

A dimension is reduced once. A second reduction over the same dimension
raises.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum

T = Set("T", np.arange(3))
m = Model("m")
x = m.var("x", (T,))
Sum(T, Sum(T, x[T]))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: term 'x' already sums over ['T']; sum over each dimension once
```

</details>
<!-- /output -->
