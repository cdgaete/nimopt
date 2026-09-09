---
title: nimblend
description: A labeled sparse N-dimensional array library, and the layer nimopt's matrices are built on.
---

# nimblend

`nimblend` is a labeled sparse N-dimensional array library. Its vocabulary is
dimensions, labels, entries and alignment, and it knows nothing about
optimization. It depends on NumPy and nothing else.

`nimopt` imports it, never the reverse. A model reaches `nimblend` in two
places: a solution is returned as a `nimblend` array, and a constraint's rows
are a `nimblend` domain. A reader who wants labeled sparse data and no model
at all can use it on its own.

## Design

**Absence is distinct from zero.** An entry is either stored or absent, and
every array declares the meaning of absence: `"empty"` for a coordinate
that contributes nothing, `"unknown"` for one that was never modelled.
Division by an absent value raises instead of producing infinity.

**One contract, two implementations.** `Array` defines what an array does.
`SparseArray` stores only the entries it has; `DenseArray` stores a grid and
the presence its declaration implies. Both are tested against the same
conformance suite.

**A coordinate is computed, not stored.** A dimension spanning millions of
positions costs nothing to hold: `ProductCoord` computes a position by
stride arithmetic and `SubsetCoord` by rank among the members of a domain.

**A domain is a set of coordinates.** It reports which members it has and
where each sits, and through `array` and `identity` it returns an array
over them, so a caller never assembles an index matrix.

## Pages

- [Arrays](/nimblend/arrays): the contract, the two implementations, the
  constructors, and what an absence declaration means.
- [Domains](/nimblend/domains): the coordinates an array has, the three ways a
  position is computed, and the buffer assembly writes into.
