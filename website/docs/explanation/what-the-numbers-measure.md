---
title: What the numbers measure
description: The benchmark figures, what each is measured against, and what none of them claims.
---

# What the numbers measure

Every figure here is given against a declared baseline. A number without its
denominator is not interpretable, and the choice of denominator changes the
size of a ratio.

## Which package allocates the bytes

Every surviving allocation of a 200x100 transport build, attributed to the
package that allocated it. Both rows are the same model over the same sets.
Rows and columns are fixed at 200 and 20 000, and the density of the
coefficient determines the nonzeros.

| nonzeros | nimblend | nimopt |
|---|---|---|
| 20 000 | 0.490 MB | 0.406 MB |
| 2 209 | 0.204 MB | 0.406 MB |

`nimblend` allocates 16.0 bytes per added nonzero, an int32 column and a
float64 value. The matrix of a model is stored in a `nimblend.EntryBuffer`,
and the CSR conversion returns views of that buffer. `nimopt` moves by
48 bytes across a ninefold change in the matrix. It allocates the per-column
vectors, the bounds and the integrality, and the row bounds. The sets fix
those counts.

**What it does not claim.** It does not claim that `nimblend` allocates more
bytes than `nimopt`. The totals follow from the nonzeros per column of the
model, and a model with one entry per column puts `nimopt` above `nimblend`.
The measure is the gradient, not the totals.

## Two models

`benchmarks/bench_transport.py` ships from plants to warehouses over a
network. Each plant serves a band of nearby warehouses, and the flow variable
spans the arcs, not the full product.

| plants | warehouses | arcs/plant | rows | columns | nonzeros | matrix | peak | ratio | build |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 100 | 10 | 300 | 2 000 | 4 000 | 0.05 MB | 0.39 MB | 8.09x | 8 ms |
| 2 000 | 500 | 20 | 2 500 | 40 000 | 80 000 | 0.96 MB | 6.95 MB | 7.24x | 45 ms |
| 10 000 | 2 000 | 40 | 12 000 | 400 000 | 800 000 | 9.60 MB | 69.71 MB | 7.26x | 314 ms |

`benchmarks/bench_storage.py` dispatches a thermal fleet, a solar fleet and a
set of batteries against an hourly demand. The batteries couple adjacent
hours through a cyclic `state_of_charge` row, and the generators through an
upward ramp limit.

| generators | batteries | hours | rows | columns | nonzeros | matrix | peak | ratio | build |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 168 | 4 862 | 2 688 | 9 724 | 0.12 MB | 0.95 MB | 8.10x | 26 ms |
| 40 | 8 | 720 | 81 320 | 46 080 | 166 960 | 2.00 MB | 14.76 MB | 7.37x | 69 ms |
| 80 | 20 | 8 760 | 2 111 080 | 1 226 400 | 4 379 840 | 52.56 MB | 370.45 MB | 7.05x | 1 552 ms |

Both models solve through HiGHS. The two lag rules are visible in the row
counts. Over 168 hours the ramp block has 10 x 167 rows: the first hour has
no predecessor, and its row is not produced. The cyclic `state_of_charge` row
wraps to the last hour and keeps all 2 x 168 of its rows.

## Peak against the matrix

The peak occurs while the model is built, not while it is assembled. On the
400 000-column transport model, the build peaks at 69.72 MB and the assembly
that follows peaks at 68.23 MB. The expression of every constraint is built
once to measure its shape, and the model that produces the matrix remains
live while the matrix is written.

The denominator determines the size of the ratio. The CSR matrix is
9.60 MB; the data a solver takes — matrix, row pointer, row and column
bounds, cost, integrality — is 21.04 MB, and the model that produced it is
21.74 MB more. Peak is 7.26x the matrix and 3.31x the full LP data, and
45.99 MB of the 69.72 MB is still live when the build returns.

The ratio is close to constant across the rungs, 8.09x at 4 000 nonzeros and
7.26x at 800 000. The figure of the smallest rung is stable and is not a
first-call artifact: three consecutive measurements in one process give 0.39,
0.38 and 0.38 MB. The excess of the small rung over the large one is the
fixed structures of the model. Those structures do not shrink with the
matrix.

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
another 8, against the 12 bytes a matrix entry occupies. An alignment
therefore costs more than the result it produces. That cost sets the lower
bound of the ratio. It is per operation and transient, and it is not
retained.

A second constraint costs its own rows and little else. Over a 500 000-cell
model, one constraint peaks at 65.71 MB against an 8.00 MB buffer, and two
constraints peak at 75.85 MB against a 16.00 MB buffer. That is 2.14 MB
beyond the rows the second constraint adds. A constraint stores its term list
and no block. An expression exists while its shape is measured and again
while it is written, and never between.

**What it does not claim.** It does not claim that peak memory is 7x the
matrix in an absolute sense. It is 7.26x *the CSR matrix* and 3.31x *the full
LP data* on this model. Which of the two applies depends on the comparison
being made.

## Against linopy

`benchmarks/bench_vs_linopy.py` builds the same three models both ways.
Inputs — the arc list, the hourly profiles, the costs — are prepared outside
the measured region and passed to both. The measured build runs from an empty
model to the matrix a solver is given: `assemble()` for nimopt, `m.matrices`
for linopy. Each side runs in a process of its own: resident memory includes
the import cost of the library that is loaded. A row is printed once the two
agree on rows, columns, nonzeros and the solved objective.

Resident memory is sampled, not traced. `tracemalloc` records only what
passes through the Python allocator. Two libraries that allocate by different
routes would then be compared on the route and not on the memory.

**A variable over a sparse subset.** The flow spans the arcs in nimopt and the
full plant-by-warehouse product under a mask in linopy.

| arcs | matrix | nimopt RSS | linopy RSS | nimopt build | linopy build |
|---|---|---|---|---|---|
| 2 000 | 0.05 MB | 3.4 MB | 27.3 MB | 3.8 ms | 177.7 ms |
| 40 000 | 0.96 MB | 10.7 MB | 107.5 MB | 27.9 ms | 206.5 ms |
| 400 000 | 9.60 MB | 72.9 MB | 1 682.9 MB | 292.9 ms | 844.5 ms |

At 400 000 arcs the mask spans a 20 000 000-cell product: nimopt builds the
same matrix in a twenty-third of the memory and a third of the time. The
design targets this case.

**Temporal coupling on a dense model.** Ramp limits and a cyclic
`state_of_charge` row over a full generator-by-hour grid.

| rows | matrix | nimopt RSS | linopy RSS | nimopt build | linopy build |
|---|---|---|---|---|---|
| 4 862 | 0.12 MB | 4.3 MB | 28.8 MB | 8.5 ms | 267.5 ms |
| 81 320 | 2.00 MB | 18.9 MB | 41.4 MB | 58.6 ms | 280.4 ms |
| 2 111 080 | 52.56 MB | 383.4 MB | 409.3 MB | 1 511.0 ms | 507.9 ms |

Here the advantage narrows and then reverses on build time. At 2 111 080 rows
linopy builds the same matrix **3.0x faster** for 7% more resident memory.
Nothing is sparse in this model, and the design costs time with no
corresponding saving. Every operation aligns by key, while xarray broadcasts
over dense grids and aligns by position. The model is also built twice, once
to measure the shape of each constraint and once to write it. Resident memory
is close on both sides: both store the same dense coefficient grids.

**Integrality.** Making the flow an integer column changes neither build: nimopt 8.6 MB and 27.3 ms against its own 10.7 MB and 27.9 ms as an LP,
linopy 108.4 MB and 204.4 ms against 107.5 MB and 206.5 ms. Integrality is a
column vector, not a matrix.

**Reading the solution back.** Primals onto their sets and duals onto their
rows: nimopt 0.7–6.9 ms across every rung, linopy 1.4–64.5 ms. The 64.5 ms is
the 400 000-arc transport model, where the solution is unpacked onto the
masked product the build used.

The matrices differ in width. linopy returns a scipy matrix with `int64`
column indices, and the same 800 000 nonzeros occupy 12.80 MB against
9.60 MB for nimopt.

**What it does not claim.** It does not claim that nimopt is faster than
linopy. On a dense temporally coupled model at two million rows it is three
times slower, and the table reports that row with the others. The numbers
support a narrower claim. Where a model is sparse in its variables, not
materialising the grid saves memory and time. Where it is not, the alignment
work costs time and saves nothing.
