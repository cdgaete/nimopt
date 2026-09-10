---
title: The package boundary
description: What belongs to nimblend, what belongs to nimopt, and the tests that keep the boundary where it is.
---

# The package boundary

Two packages, with the dependency in one direction. `nimopt` imports
`nimblend`; `nimblend` never imports `nimopt`.

**`nimblend`** is a labeled sparse N-dimensional array. Its vocabulary is
dimensions, labels, entries and alignment. It contains no optimization term.
A function in it referring to a row, a column or a constraint is a boundary
violation.

**`nimopt`** is an LP/MILP builder in which a variable is a dimension. Its
types are `nimblend` arrays with names attached, and its matrix is stored in
a `nimblend.EntryBuffer`.

The boundary is not a convention. It is enforced by tests that fail when it
moves.

## What the tests enforce

**`nimblend` never refers to `nimopt`.** A scan of the `nimblend` source
fails on any mention. A second test imports `nimblend` alone and fails where
`nimopt` is imported with it. `nimblend` enforces the other half of the rule
in its own suite: no class, function or parameter it declares is named for a
constraint, an objective, a solver or a variable, and no source file mentions
one in prose. A package developed on its own fails in its own suite, and not
in the suite of a consumer.

**Every `nimblend` import is a public name of the top-level module.** A model
imports `SparseArray` from `nimblend`, never from `nimblend.sparse`. Importing a
public name by its submodule path is how a dependency on an internal
starts. The modules `nimopt` ships are constrained further: the set of
`nimblend` names they import is pinned, and widening it requires an explicit
change to the test.

**No array's `.index` or `.data` is read.** Those are the raw index matrix
and value buffer of the layer below the array layer. The rule is checked by
walking the syntax tree, not by a text search. It detects a read that is not
a subscript, and it permits `dims.index(name)`, a position lookup on a
tuple.

**And none is assembled.** Reading a buffer is one half of the bypass, and
building one is the other. An index matrix assembled in `nimopt` is array
work performed one layer too high. That work is part of the interface that
moves when the kernel below `nimblend` is replaced. A domain returns the array over
its own members instead: `array(values)` assigns a value to each member,
and `identity(into, coord, start)` pairs each with its position along a new
dimension. No module of `nimopt` calls `SparseArray(index, ...)` or
`from_canonical`.

## Where the bytes are allocated

The strongest boundary test measures allocation. It does not compare the
totals of the two packages.

The cost per nonzero of a build is allocated in `nimblend`: the matrix is an
int32 column and a float64 value per entry, stored in a `nimblend` buffer. A
ninefold increase in the nonzeros grows the `nimblend` share by at least
twelve bytes per added entry.

`nimopt` allocates a near-constant amount across the same change. What it
allocates is what a solver takes beside the matrix: a lower bound, an upper
bound and an integrality flag per column, and a pair of bounds per row. The
sets fix those counts, and the density does not.

The test therefore measures the gradient and not the totals. An assertion
that `nimblend` allocates more bytes would measure the workload instead of
the boundary. `nimopt` allocates a vector per column at every density, and a
sparse enough model puts `nimopt` above `nimblend` with no boundary
violation.

## Why the site lives in `nimopt`

`nimopt` imports `nimblend`. A site inside `nimopt` documenting both packages
follows that dependency. A site inside `nimblend` documenting `nimopt` would
invert it, and the `nimblend` tests would then require a model to describe.
