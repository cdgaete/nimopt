---
title: The package boundary
description: What belongs to nimblend, what belongs to nimopt, and the tests that keep the boundary where it is.
---

# The package boundary

Two packages, with the dependency in one direction. `nimopt` imports
`nimblend`; `nimblend` never imports `nimopt`.

**`nimblend`** is a labeled sparse N-dimensional array. Its vocabulary is
dimensions, labels, entries and alignment. It knows nothing about
optimization, and a function in it referring to a row, a column or a
constraint would be a boundary violation.

**`nimopt`** is an LP/MILP builder in which a variable is a dimension. Its
types are `nimblend` arrays with names attached, and the buffer its matrix
lives in is a `nimblend.EntryBuffer`.

The boundary is not a convention. It is enforced by tests that fail when it
moves.

## What the tests enforce

**`nimblend` never refers to `nimopt`.** A scan of `nimblend`'s source fails on any
mention. A second test imports `nimblend` alone and fails if `nimopt` is
imported with it. `nimblend` carries the other half of the rule in its own
suite: no class, function or parameter it declares is named for a
constraint, an objective, a solver or a variable, and no source file of it
mentions one even in prose. A package developed on its own needs its own
suite to fail, rather than waiting for a consumer's.

**Every `nimblend` import is a public name of the top-level module.** A model
imports `SparseArray` from `nimblend`, never from `nimblend.sparse`. Importing a
public name by its submodule path is how a dependency on an internal
starts. The modules `nimopt` ships are held tighter still: the set of `nimblend`
names they import is pinned, so widening it is a deliberate act rather than
drift.

**No array's `.index` or `.data` is read.** Those are the raw index matrix
and value buffer of the layer below the array layer. The rule is checked by
walking the syntax tree rather than by grepping, so it catches a read that
is not a subscript, and it allows `dims.index(name)`, which is a tuple
being asked for a position.

**And none is assembled.** Reading a buffer is one half of the bypass and
building one is the other. An index matrix assembled in `nimopt` is array
work done a layer too high, and it is exactly the seam that has to move
when the kernel below `nimblend` is replaced. A domain returns the array over
its own members instead: `array(values)` assigns a value to each member,
and `identity(into, coord, start)` pairs each with its position along a new
dimension. No module of `nimopt` calls `SparseArray(index, ...)` or
`from_canonical`.

## Where the bytes land

The sharpest boundary test concerns allocation, and it does not ask which
package holds more.

A build's cost per nonzero lands in `nimblend`: the matrix is an int32 column
and a float64 value per entry, and it lives in a `nimblend` buffer. Increase
the nonzeros ninefold and `nimblend`'s share grows by at least twelve bytes
for each one added.

`nimopt` stays flat across the same change. What it allocates is what a
solver takes beside the matrix: a lower bound, an upper bound and an
integrality flag per column, and a pair of bounds per row. Those are fixed
by the sets, not by the density.

That is why the test is written as a gradient rather than a comparison.
Asserting that `nimblend` simply holds more bytes would measure the workload
instead of the boundary: `nimopt` holds a vector per column whatever the
density, so a sparse enough model puts `nimopt` above `nimblend` with nothing
having drifted.

## Why the site lives in `nimopt`

`nimopt` imports `nimblend`, so a site inside `nimopt` documenting both runs
with the dependency. A site inside `nimblend` documenting `nimopt` would invert
it, and the inversion would be real: `nimblend`'s own tests would need a model
to describe.
