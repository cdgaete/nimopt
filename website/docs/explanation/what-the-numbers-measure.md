---
title: What the numbers measure
description: The benchmark figures, what each is measured against, and what none of them claims.
---

# What the numbers measure

Every figure here is given against a stated baseline. A number without its
denominator says nothing, and a ratio can be made to sound like anything by the choice of denominator.

## Who owns the bytes

Every surviving allocation of a 200x100 transport build, attributed to the package that allocated it. Both rows are the same model over the same sets, so
rows and columns are fixed at 200 and 20 000, and the coefficient's density
alone decides the nonzeros.

| nonzeros | nimblend | nimopt |
|---|---|---|
| 20 000 | 0.490 MB | 0.406 MB |
| 2 209 | 0.204 MB | 0.406 MB |

`nimblend` absorbs 16.0 bytes per added nonzero — an int32 column and a float64
value — because the buffer a model's matrix lives in is a `nimblend.EntryBuffer`
and the CSR handoff returns views of it. `nimopt` moves by 48 bytes across a
ninefold change in the matrix: what it allocates is the per-column vectors,
bounds and integrality, and the row bounds, which the sets fix.

**What it does not claim.** Not that `nimblend` holds more bytes than `nimopt`.
That is a property of a model's nonzeros per column, and a model carrying one
entry per column puts `nimopt` above `nimblend` with nothing having drifted. The
measure is the gradient, not the totals.

## Two models

`benchmarks/bench_transport.py` ships from plants to warehouses over a
network in which each plant serves a band of nearby warehouses, so the flow
variable spans the arcs rather than the full product.

| plants | warehouses | arcs/plant | rows | columns | nonzeros | matrix | peak | ratio | build |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 100 | 10 | 300 | 2 000 | 4 000 | 0.05 MB | 0.39 MB | 8.09x | 8 ms |
| 2 000 | 500 | 20 | 2 500 | 40 000 | 80 000 | 0.96 MB | 6.95 MB | 7.24x | 45 ms |
| 10 000 | 2 000 | 40 | 12 000 | 400 000 | 800 000 | 9.60 MB | 69.71 MB | 7.26x | 314 ms |

`benchmarks/bench_storage.py` dispatches a thermal fleet, a solar fleet and a
set of batteries against an hourly demand. The batteries couple neighbouring
hours through a cyclic state of charge, and the generators through an upward
ramp limit.

| generators | batteries | hours | rows | columns | nonzeros | matrix | peak | ratio | build |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 168 | 4 862 | 2 688 | 9 724 | 0.12 MB | 0.95 MB | 8.10x | 26 ms |
| 40 | 8 | 720 | 81 320 | 46 080 | 166 960 | 2.00 MB | 14.76 MB | 7.37x | 69 ms |
| 80 | 20 | 8 760 | 2 111 080 | 1 226 400 | 4 379 840 | 52.56 MB | 370.45 MB | 7.05x | 1 552 ms |

Both solve through HiGHS. The two lag rules are visible in the row counts: over 168 hours the ramp block has 10 x 167 rows, because the first hour has no predecessor and its row is not produced, while the cyclic state of charge wraps to the last hour and keeps all 2 x 168 of its rows.

## Peak against the matrix

The peak is reached while the model is built, not while it is assembled. On
the 400 000-column transport model, building peaks at 69.72 MB and the
assembly that follows peaks at 68.23 MB, because every constraint's
expression is built once to measure its shape and the model that produces the
matrix stays live while the matrix is written.

The denominator decides how large the ratio sounds. The CSR matrix is
9.60 MB; the data a solver takes — matrix, row pointer, row and column
bounds, cost, integrality — is 21.04 MB, and the model that produced it is
21.74 MB more. Peak is 7.26x the matrix and 3.31x the full LP data, and
45.99 MB of the 69.72 MB is still live when the build returns.

The ratio is close to flat across the rungs, 8.09x at 4 000 nonzeros and
7.26x at 800 000, and the smallest rung's figure is stable rather than a
first-call artefact: three consecutive measurements of it in one process give
0.39, 0.38 and 0.38 MB. What the small rung has above the large is the
model's own fixed structures, which do not shrink with the matrix.

Where the transient bytes go, attributed by the frame that allocated them at
the moment the 400 000-column build peaks:

| bytes | allocated by |
|---|---|
| 16.26 MB | the caller's own arc labels and costs |
| 14.40 MB | the broadcast product of a parameter against a variable |
| 12.80 MB | the sorted copies the two unordered blocks need |
| 8.00 MB | the variable's own index, values and column coordinate |
| 6.40 MB | the subset's codes and its index in code order |
| 3.20 MB | the positions a probe resolves to |
| 3.20 MB | a ravelled key set |

An `int64` ravel key costs 8 bytes per entry and an argsort permutation
another 8, against the 12 bytes a matrix entry finally occupies, so an
operation that aligns costs more than the result it produces. That is the
floor the ratio rests on, and it is per-operation and transient rather than
retained.

A second constraint costs its own rows and little else. Over a 500 000-cell
model, one constraint peaks at 65.71 MB against an 8.00 MB buffer and two
peak at 75.85 MB against a 16.00 MB buffer: 2.14 MB beyond the rows the
second adds, because a constraint holds its term list rather than a block, so
an expression exists while its shape is measured and again while it is
written, and never between.

**What it does not claim.** Not that peak memory is 7x the matrix in any
absolute sense. It is 7.26x *the CSR matrix* and 3.31x *the full LP data* on
this model, and which of those a reader cares about depends on what they were
going to compare it with.

## Against linopy

`benchmarks/bench_vs_linopy.py` builds the same three models both ways.
Inputs — the arc list, the hourly profiles, the costs — are prepared outside
the measured region and handed to both. The measured build runs from an empty
model to the matrix a solver would be given: `assemble()` for nimopt,
`m.matrices` for linopy. Each side runs in a process of its own, because
resident memory carries the import cost of whichever library is loaded, and a
row is printed only once the two agree on rows, columns, nonzeros and the
solved objective.

Resident memory is sampled rather than traced. `tracemalloc` sees only what
passes through Python's allocator, and two libraries that allocate memory by different routes would be compared on the route rather than on the memory.

**A variable over a sparse subset.** The flow spans the arcs in nimopt and the
full plant-by-warehouse product under a mask in linopy.

| arcs | matrix | nimopt RSS | linopy RSS | nimopt build | linopy build |
|---|---|---|---|---|---|
| 2 000 | 0.05 MB | 3.4 MB | 27.3 MB | 3.8 ms | 177.7 ms |
| 40 000 | 0.96 MB | 10.7 MB | 107.5 MB | 27.9 ms | 206.5 ms |
| 400 000 | 9.60 MB | 72.9 MB | 1 682.9 MB | 292.9 ms | 844.5 ms |

At 400 000 arcs the mask spans a 20 000 000-cell product: nimopt builds the
same matrix in a twenty-third of the memory and a third of the time. This is
the axis the design is for.

**Temporal coupling on a dense model.** Ramp limits and a cyclic state of
charge over a full generator-by-hour grid.

| rows | matrix | nimopt RSS | linopy RSS | nimopt build | linopy build |
|---|---|---|---|---|---|
| 4 862 | 0.12 MB | 4.3 MB | 28.8 MB | 8.5 ms | 267.5 ms |
| 81 320 | 2.00 MB | 18.9 MB | 41.4 MB | 58.6 ms | 280.4 ms |
| 2 111 080 | 52.56 MB | 383.4 MB | 409.3 MB | 1 511.0 ms | 507.9 ms |

Here the advantage narrows and then reverses on time: at 2 111 080 rows
linopy builds the same matrix **3.0x faster** for 7% more resident memory.
Nothing is sparse in this model, so what remains is what the design costs where it gains nothing — every operation aligns by key, while xarray
broadcasts over dense grids and aligns by position — and the model is built
twice, once to measure each constraint's shape and once to write it. Resident
memory stays close because both end up holding the same dense coefficient
grids.

**Integrality.** Making the flow an integer column moves neither side's
build: nimopt 8.6 MB and 27.3 ms against its own 10.7 MB and 27.9 ms as an LP,
linopy 108.4 MB and 204.4 ms against 107.5 MB and 206.5 ms. Integrality is a
column vector, not a matrix.

**Reading the answer back.** Primals onto their sets and duals onto their
rows: nimopt 0.7–6.9 ms across every rung, linopy 1.4–64.5 ms, the 64.5 ms
being the 400 000-arc transport model, where the answer is unpacked onto the same masked product the build used.

The matrices differ in width: linopy hands over a scipy matrix with `int64`
column indices, so the same 800 000 nonzeros occupy 12.80 MB against nimopt's
9.60 MB.

**What it does not claim.** Not that nimopt is faster than linopy. On a dense
temporally-coupled model at two million rows it is three times slower, and
that row is in the table for the same reason the others are. The claim the
numbers support is narrower: where a model is sparse in its variables, not
materialising the grid is worth a great deal, and where it is not, the alignment work is a cost with no corresponding saving.
