---
title: For agents
description: The mental model, the public surface, and the mistakes to avoid when writing nimopt.
---

# For agents

The whole manual is at [`/llms-full.txt`](/llms-full.txt); the index is at
[`/llms.txt`](/llms.txt).

## The mental model

**A variable is a dimension.** `m.var("x", (P, W))` occupies a block of the
model's single column space. A member's column is computed from its
multi-index and is not stored. A variable over millions of columns costs
its members, not its columns.

**An expression is symbolic.** `cost[P, W] * x[P, W]` contains references,
not arrays. Writing it allocates nothing. It becomes matrix entries when a
constraint is materialised.

**A constraint is an array.** It is a `nimblend` array over its free sets
crossed with the column space. There is no assembly step: the array is the
matrix.

**A definition is a model without its data.** `Definition` provides the
vocabulary a model is written in, `set`, `param`, `var`, `constraint` and
`set_objective`, over symbols declared with no members and no values.
`explain()` reports what it declares. `build(data)` binds a copy and returns
a `Model`. One definition builds a model for each dataset it is given.

**A built model is inspected through three methods.** `explain()` reports
what it built, and `row(name, **coords)` reads one row out of the assembled
matrix. `absent(name)` reports which coordinates were dropped from a
constraint and by which rule. All three read the assembled model, and none
walks the expression a second time.

**A session keeps the solver open.** `model.session()` assembles once and
keeps the solver's model. `diagnose()` queries the solved instance for the
conflicting rows of an infeasible model, or for the ray of an unbounded one.
`available()` lists the installed adapters. `capabilities(name)` reports what
one adapter supports and which capabilities it rejects together. A model with
integer columns has no duals: a mixed-integer model's duals are not its
relaxation's.

**`nimblend` is the layer below.** Its vocabulary is dimensions, labels,
entries and alignment, and it contains no optimization term. Import from
`nimblend` itself, never from `nimblend.sparse` or another submodule, and
never read an array's `.index` or `.data` or a domain's `.codes`. Each of the
three has a reader above it:
`coordinates()`, `values()`, `positions_of_coordinates()` and `as_coord()`.
Do not build an index matrix either: a domain returns the array over its own
members through `array(values)` and `identity(into, coord, start)`.

## The public surface

<!-- surface -->
| From | Names |
| --- | --- |
| `nimopt` | `COLUMN`, `ROW`, `Absence`, `Alias`, `Assembled`, `Coefficient`, `Constraint`, `Definition`, `Diagnosis`, `Explanation`, `Expression`, `Model`, `Option`, `Param`, `Relation`, `Row`, `Session`, `Set`, `Solution`, `Sum`, `Term`, `Variable`, `available`, `capabilities`, `load`, `loads`, `options`, `product`, `save`, `subset`, `subset_of` |
| `nimblend` | `Array`, `DenseArray`, `Domain`, `EntryBuffer`, `SparseArray`, `combined_dims`, `from_long`, `from_dense`, `is_canonical`, `StoredCoord`, `ProductCoord`, `SubsetCoord` |
<!-- /surface -->

**A coefficient composes.** A coefficient is a parameter read at its sets
or an arithmetic combination of such readings: `price[G, T] / eta[G, T]` is
a coefficient written before any data exists, read at its sets like a
parameter, and evaluated once when the matrix is built. `+`, `-`, `*`, `/`
and a power by a number combine coefficients. An expression also contains a
constant: `x + 1 <= 5` produces the row `x <= 4`.

## What goes wrong

**A chained comparison.** `0 <= expr <= 10` raises `TypeError`. Python
evaluates it as two comparisons joined by `and` and keeps only the second.
A relation has no truth value, and the chained form raises instead of
dropping the first bound. Write each bound as its own constraint.

**A sum over a lag.** `Sum(T - 1, ...)` raises: a sum runs over a set's
members. Put the lag on the variable reference, `x[T - 1]`.

**The built-in `sum` over a set's members.** `sum(x[S, t] for t in members)`
returns the correct expression at a cost: it produces one term per member,
where `Sum(T, x[S, T])` produces one term and reduces a dimension. The terms
concatenate pairwise and each materialises its own block. Building a model
that way runs 24 times slower at 25 members and 275 times slower at 400, and
the factor grows with the member count. Use the built-in `sum` for a short
list of distinct expressions and `Sum` for a set's members.

**A right-hand side over the wrong dimensions.** A constraint's right-hand
side is a parameter over exactly its free dimensions. The error message
gives both.

**Reading values from a model that did not solve.** `objective` and
`primal` raise where `feasible` is False. `dual` raises where `status` is
not `"optimal"`. A solve stopped at a limit reports `feasible` True where
the solver found a point, with `bound` and `gap` beside it. Read `status`
first.

**A domain over a definition's sets.** `product((B, T))` needs each set's
coordinate, and a declared set has none. In a definition, give `where=`,
`over=` and `subset=` as a tuple of its sets or as one of its parameters,
whose coefficients are the coordinates.

**A row that is not there.** `row()` raises for a coordinate at which the
constraint has no row. `absent()` reports which rule dropped it: a
coefficient absent inside a sum removes a **term** and keeps the row; a term
absent along a **free** dimension removes the **row**.

**Reading a MILP's duals.** A model with integer columns has no duals.
`dual()` raises; it does not return the relaxation's duals. Read `primal`.

**A conflict HiGHS cannot prove.** HiGHS computes its conflict over the
linear relaxation. A model infeasible only through its integrality produces
no conflict, and `diagnose()` raises. The Gurobi conflict covers the
integrality.

**Two operands that share no dimension.** Every binary operator combines two
dimensioned operands only where they share a dimension. The rule applies to a
coefficient multiplied by a variable and to two coefficients alike. Frames
sharing no dimension raise `ValueError`; their combination would be an outer
product. A number has no dimension and scales every entry.

**A division by zero.** A divisor that is zero raises `ZeroDivisionError`
with the coordinate, for a Python number, a NumPy scalar and a coefficient
with a zero at one coordinate alike. Handle the divisor before it is passed
to an expression.

**A derived coefficient read at the wrong sets.** A combination is read at
its sets as a parameter is, and the reading is checked against the
dimensions it has. `unit_cost[T, G]` raises where it is written and reports
`('G', 'T')`.

**Bypassing the `nimblend` interface.** A test fails on an import from a
`nimblend` submodule. It also fails on a read of an array's `.index` or
`.data` or a domain's `.codes`, and on a module of the package that assembles
an index matrix of its own.

## Every error, and where it is shown

The prose above covers the common mistakes. The table lists every error the
documentation demonstrates. Each row is executed to produce the message
beside it.

<!-- refusals -->
| Raises | Message | Shown at |
| --- | --- | --- |
| `ValueError` | the upper bound 'cap' has no value at member ('b',) of variable 'x'; a bound covers every column of the variable it bounds | [/guides/bounds-from-parameters](/guides/bounds-from-parameters) |
| `ValueError` | variable 'x' is declared over ('G',) and is not over ['W']; its upper bound 'cap' is declared over ('W',) | [/guides/bounds-from-parameters](/guides/bounds-from-parameters) |
| `TypeError` | a coefficient is a parameter; build one with `Param.from_dense` or `Param.from_long` and read it at its sets. A product of two expressions is not linear. | [/guides/coefficient-arithmetic](/guides/coefficient-arithmetic) |
| `ValueError` | coefficient (fuel_price / efficiency) is over ('G', 'T'); got ('T', 'G') | [/guides/coefficient-arithmetic](/guides/coefficient-arithmetic) |
| `ZeroDivisionError` | divisor holed is zero at 1 coordinate(s), first at {'G': 'base', 'T': 1}; remove the zeros or divide by another parameter | [/guides/coefficient-arithmetic](/guides/coefficient-arithmetic) |
| `ValueError` | frames ('G',) and ('T',) share no dimension; there is nothing to align them on | [/guides/coefficient-arithmetic](/guides/coefficient-arithmetic) |
| `ValueError` | constraint 'capacity' has free dimensions ('P',); its condition is over ('W',) | [/guides/conditions](/guides/conditions) |
| `ValueError` | constraint 'capacity' is given over= and where= together; pass one of them | [/guides/conditions](/guides/conditions) |
| `ValueError` | variable 'x' is read at member 't9' of dimension 'T'; read it at a member that set contains | [/guides/fixed-members](/guides/fixed-members) |
| `ValueError` | a lag is a whole number of members; got 1.7 | [/guides/lags](/guides/lags) |
| `ValueError` | a sum is over the members of ['T'] and takes the set, not a lag of it; write the lag at the variable's reference | [/guides/lags](/guides/lags) [/reference/expression](/reference/expression) |
| `ValueError` | parameter 'rate' is read at a lag ['T']; write the lag at the variable's reference | [/guides/lags](/guides/lags) |
| `ValueError` | 'max(gen[G, T]) <= 10': the syntax supports one call; write Sum | [/guides/saving-and-loading](/guides/saving-and-loading) |
| `ValueError` | capital does not fall from base to what follows it; pass a capital cost that falls across the merit order | [/models/expansion](/models/expansion) |
| `ValueError` | frames ('P',) and ('Q',) share no dimension; there is nothing to align them on | [/nimblend/arrays](/nimblend/arrays) |
| `ValueError` | label column 't' has length 2 and the value column has length 1; they name the same entries | [/nimblend/arrays](/nimblend/arrays) |
| `ValueError` | this array declares absence 'unknown' and does not carry every coordinate of its frame, so densifying must state fill=<value> to place at the rest | [/nimblend/arrays](/nimblend/arrays) [/tutorial/reading-the-answer](/tutorial/reading-the-answer) |
| `ValueError` | 3 member(s) numbered from 4 reach position 6, and dimension 'k' spans 6 | [/nimblend/domains](/nimblend/domains) |
| `ValueError` | a domain of 3 member(s) takes one value each, as a column of that length; got shape (2,) | [/nimblend/domains](/nimblend/domains) |
| `ValueError` | constraint 'supply' has free dimensions ('P',); its right-hand side 'demand' is over ('W',) | [/reference/constraint](/reference/constraint) [/tutorial/constraints](/tutorial/constraints) |
| `ValueError` | data does not cover ['S']; add an entry for each | [/reference/definition](/reference/definition) |
| `ValueError` | parameter 'S' is already declared as a set; declare another name | [/reference/definition](/reference/definition) |
| `TypeError` | a relation has no truth value; write each bound in its own equation | [/reference/expression](/reference/expression) [/tutorial/constraints](/tutorial/constraints) |
| `TypeError` | a relation is already an equation with one bound; compare the expression again in its own equation | [/reference/expression](/reference/expression) |
| `TypeError` | an LP has no row for a strict inequality; write `<=` or `>=`, and reduce with `Sum` in place of `min` or `max` | [/reference/expression](/reference/expression) |
| `TypeError` | an expression has no absolute value: expressions are linear; bound the expression with two rows, or reduce it with `Sum` over its sets | [/reference/expression](/reference/expression) |
| `TypeError` | an expression is reduced over the sets it is summed across; specify them with `Sum(I, J, expression)` | [/reference/expression](/reference/expression) |
| `TypeError` | cannot divide by an expression: expressions are linear; declare the reciprocal as a coefficient the variable multiplies | [/reference/expression](/reference/expression) |
| `TypeError` | cannot raise an expression to a power: expressions are linear; raise a coefficient to the power and multiply it by a variable | [/reference/expression](/reference/expression) |
| `ValueError` | term 'x' already sums over ['T']; sum over each dimension once | [/reference/expression](/reference/expression) |
| `ValueError` | constraint 'cap' gives where= a domain with no name; declare its members as a parameter and refer to that parameter | [/reference/files](/reference/files) |
| `ValueError` | parameter 'c' is given columns ['value', 'S']; a table lists the dimensions then value: ['S', 'value'] | [/reference/files](/reference/files) |
| `ValueError` | variable 'x' contains the unknown keys ['bound']; write only ('sets', 'subset', 'lower', 'upper', 'integer') | [/reference/files](/reference/files) |
| `ValueError` | constraint 'cap' has no row at {'P': 'p3'}; read `absent('cap')` for the rule that dropped it | [/reference/inspection](/reference/inspection) |
| `ValueError` | parameter 'cost': label column 'P' has length 1 and the value column has length 2; they name the same entries | [/reference/param](/reference/param) |
| `TypeError` | parameter 'price' is over ('G',) and expresses no coefficient until it is read; read it at its sets as price[G] | [/reference/param](/reference/param) |
| `ValueError` | status is 'infeasible' and the solver reports no feasible point; read `status` before reading values | [/reference/solution](/reference/solution) [/tutorial/solving](/tutorial/solving) |
| `ValueError` | model 'm' has integer columns and 'highs' reports no duals for it; read primal values only | [/reference/solvers](/reference/solvers) |
| `TypeError` | parameter 'supply' is over ('P',) and expresses no coefficient until it is read; read it at its sets as supply[P] | [/tutorial/constraints](/tutorial/constraints) |
| `ValueError` | parameter 'cost' is over sets of shape (2, 3); got values of shape (2, 2) | [/tutorial/sets-and-parameters](/tutorial/sets-and-parameters) |
<!-- /refusals -->
