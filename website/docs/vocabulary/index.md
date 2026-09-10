---
title: Vocabulary
description: The terms used throughout the documentation, each defined once.
---

# Vocabulary

Terms used throughout the documentation, each defined once. Examples refer
to the transport model: plants `P = {lisbon, porto}` ship to warehouses
`W = {berlin, paris, rome}`.

## Index sets

**Set.** A named index dimension with labels. `Set("P", np.array(["lisbon",
"porto"]))` is the set of plants. Parameters, variables and constraints are
indexed over sets, and solution values are returned over the same sets.

**Member.** One element of a set. `"lisbon"` is a member of `P`.

**Label.** The name of a member, a string or a number. Labels are the
caller-facing identifiers; integer positions are used internally.

**Set product.** The Cartesian product of several sets. `P × W` has six
members, `("lisbon", "berlin")`, `("lisbon", "paris")` and so on. Variables
and parameters are indexed over set products.

**Subset.** An explicit list of members of a set product. `subset((P, W),
{"P": ..., "W": ...})` lists the routes that exist. A variable over a subset
has a column per listed member and none for the rest.

**Alias.** A second name for a set, sharing its labels. It allows a
parameter or a constraint to relate a set to itself, such as a flow between
two nodes of one node set.

**Domain.** A set of coordinates over some dimensions. `product((P, W))` and
`subset(...)` return one. `subset=`, `where=` and `over=` take a domain.

## Data and decisions

**Parameter.** Data indexed over a set product: one value per member.
`cost` is indexed over `(P, W)`; `supply` over `P`. A parameter has no
column in the matrix.

**Coefficient.** The multiplier of a variable in a row. A parameter indexed
at its sets, `cost[P, W]`, is a coefficient, as is an arithmetic
combination of such readings, `price[G, T] / eta[G, T]`.

**Variable.** A decision variable. `m.var("x", (P, W))` declares one
decision per route. Values are read after a solve with `primal("x")`.

**Column.** One decision in the coefficient matrix. Each member of a
variable is one column. Column indices are computed from member positions
and are never assigned by the caller.

**Bound.** The interval a column may take values in. The default lower
bound is 0 and the default upper bound is infinity.

## Expressions

**Expression.** A linear combination of variables, such as `Sum(W, x[P,
W])`. Writing an expression records its structure and computes nothing.
Values are read when the matrix is built.

**Term.** One component of an expression: one variable, an optional
coefficient, and the sets summed over. `cost[P, W] * x[P, W]` is one term.

**Frame.** The dimensions an expression is still indexed over, also called
its free dimensions. `x[P, W]` has frame `(P, W)`; `Sum(W, x[P, W])` has
frame `(P,)`. An empty frame is a scalar.

**Sum.** Summation over the members of the named sets. The summed sets are
removed from the frame.

**Lag.** A reference to the previous or next member of a set. `x[T - 1]`
references the previous period. A lag either drops the row with no
predecessor or, with `T.cyclic`, wraps to the last member.

**Fixed member.** A label in place of a set in a reference, `x[G, "t0"]`.
It selects that member and removes the set from the frame.

## Constraints and the matrix

**Relation.** An expression compared with `<=`, `>=` or `==` to a
right-hand side. `Sum(W, x[P, W]) <= supply[P]` is a relation. It becomes
of the model when passed to `m.constraint`.

**Constraint.** A relation added to the model under a name. It produces one
row per member of its expression's frame.

**Row.** One inequality or equality of the coefficient matrix. The supply
constraint over two plants produces two rows.

**Right-hand side.** The scalar or parameter on the other side of the
relation. A scalar applies to every row. A parameter must be indexed over
exactly the constraint's frame, so that each row has its own value.

**Objective.** A scalar expression, one with an empty frame, that the solver
minimizes or maximizes. `Sum(P, W, cost[P, W] * x[P, W])` is the total
shipping cost.

**Sense.** The optimization direction, `"min"` or `"max"`, set once on the
`Model`.

**Materialise.** Evaluate a parameter or an expression into an array of
values. Materialisation runs when the matrix is built, not when the
expression is written.

**Assemble.** Build the coefficient matrix from every constraint. `solve()`
assembles before calling the solver. `assemble()` returns the matrix without
solving.

**Nonzero.** One stored coefficient of the matrix. `nnz` is the count.

## Solutions

**Solution.** The return value of `solve()`: a status, an objective value,
and primal and dual values.

**Status.** The outcome the solver reported: `optimal`, `infeasible`,
`unbounded`, or a limit reached. Values are defined only for `optimal`.

**Primal.** The value of a variable in the solution, returned over the sets
it was declared on.

**Dual.** The dual value of a constraint, also called the shadow price: the
change in the objective per unit change in that row's right-hand side.
Returned over the constraint's frame.

**Absence.** A coordinate at which an array has no value, as distinct from
a stored zero. Every array declares the meaning of absence: `"empty"` for a
coordinate that contributes nothing, used by parameters, or `"unknown"` for
one that was never modeled, used by solutions.

**Session.** A solver instance kept open on one assembled model. A caller
queries it after the solve, for the conflicting rows of an infeasible
model.

**Definition.** A model written before its data exists, in the same
vocabulary. `build(data)` produces a `Model` for one dataset.
