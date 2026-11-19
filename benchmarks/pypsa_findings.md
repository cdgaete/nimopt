# What the European network costs each PyPSA backend

`RESULTS-EU-interconnected-transport.nc`, a sector-coupled European capacity
expansion network: 485 buses, 2920 snapshots, 469 generators, 71 lines, 1911
links, 278 stores, 56 storage units. PyPSA builds its model through
`n.optimize.create_model` on the `nimopt` backend and on the `linopy` backend.
The `pypsa` rung of `bench_vs_linopy.py` measures both, each in a process of
its own. At both horizons measured, the two backends build matrices with the
same rows, columns and nonzeros. The nimopt side of that rung builds with
`singleton_rows="rows"` and writes every declared row. Where the problem was
solved, both solve to the same objective.

## Setup

An AMD Ryzen 7 5800X with 8 cores, 16 threads and 62 GB of memory. highspy
1.15.1, linopy 0.9.1 and numpy 2.5.2. PyPSA is the `nimopt-backend` branch, 16
commits after v1.4.0, and reports version 1.3.0. nimopt and nimblend at the
commit of this file. Every run sets `PYTHONHASHSEED=0`. Each number is one
run. The one-minute load average read between 3.5 and 4.4 at the start and
the end of the runs of the whole year, while another process ran on the
machine. It was not read for the run of twenty-four snapshots. Across three
runs the nimopt build of the whole year took between 11.4 s and 11.7 s, and
the linopy build between 10.2 s and 10.6 s.

Each phase contains the following work:

- **build**: `n.optimize.create_model` on the backend, and the matrix. The
  nimopt side calls `assemble()` to report the shape. The linopy side reads
  `model.matrices.A`. The network file is read before the phase starts.
- **solve**: `model.solve` with HiGHS and its default method. linopy
  passes the matrix in memory with `io_api="direct"`. The nimopt session
  assembles the matrix a second time before it passes it to HiGHS.
- **read**: `assign_solution`, `assign_duals` and `post_processing` on the
  network.

The linopy column count excludes the column of the objective constant, which
has no matrix entry.

## Twenty-four snapshots, solved on both backends

162,582 rows by 79,457 columns, 382,328 nonzeros, 4.81 per column, 2,314 dense
columns. Both objectives are 232,416,019,336.75.

| | nimopt | linopy |
| --- | --- | --- |
| build | 892.7 ms | 2,820.2 ms |
| matrix | 4.6 MB | 6.1 MB |
| solve | 243.0 s | 505.9 s |
| read | 4.8 s | 4.8 s |
| resident added by the solve | 382.0 MB | 523.7 MB |
| peak resident of the solve | 819.2 MB | 954.8 MB |
| peak resident of the process | 819.2 MB | 957.3 MB |

The generator dispatch sums to 118,174,790.05 on both backends. The nodal
prices sum to 10,637,906.68 on the nimopt backend and to 10,688,675.86 on the
linopy backend. The two solves return different dual points.

## The whole year, built on both backends

19,423,878 rows by 9,361,137 columns, 46,354,588 nonzeros, 4.95 per column,
2,391 dense columns.

| | nimopt | linopy |
| --- | --- | --- |
| build | 11.4 s | 10.2 s |
| matrix | 556 MB | 742 MB |
| resident added by the build | 3,202 MB | 4,747 MB |
| peak resident of the process | 3,937 MB | 5,513 MB |

A separate nimopt run times the two parts of the build: `create_model` takes
6.5 s and `assemble` takes 4.7 s.

## The whole year under both values of `singleton_rows`

`bench_pypsa_backend.py` measures the nimopt backend under both values. Under `"bounds"`, the default, 8,801,358 of the 19,423,878 declared rows are stored as column bounds, and the solver loads 10,622,520 rows and 37,553,230 nonzeros. Under `"rows"` the solver loads every declared row.

| | `"rows"` | `"bounds"` |
| --- | --- | --- |
| loaded rows | 19,423,878 | 10,622,520 |
| matrix | 556.3 MB | 450.6 MB |
| build | 11.2 s to 11.6 s | 11.5 s to 11.8 s |
| resident added by the build | 3,180 MB to 3,185 MB | 2,902 MB to 2,905 MB |
| peak resident of the process in the build | 3,915 MB to 3,917 MB | 3,636 MB to 3,637 MB |
| solve | 1,597 s | 1,575 s |
| resident added by the solve | 40,059 MB | 36,368 MB |
| peak resident of the process in the solve | 42,190 MB | 38,534 MB |
| read | 6.4 s | 6.2 s |

The build is three runs of each value, in alternation. nimopt 0.20261118.1, in the same alternation, builds in 11.1 s to 11.3 s with a peak of 3,932 MB to 3,935 MB. The solve is one run of each value with Mosek 11.2, barrier without crossover, at 16 threads. Both solves return status `optimal` and the objective 1,058,936,103,870.507. Mosek reports 6,457,581 constraints and 12,095,142 variables after its presolve for both matrices, and 4.82e8 nonzeros in its factor for both.

The solve under `"bounds"` peaks 3,656 MB lower. The build under `"bounds"` peaks 278 MB to 280 MB lower and takes 0.2 s to 0.5 s longer.

## What the numbers show

**The nimopt build is faster at twenty-four snapshots and slower over the
year.** It takes 0.89 s against 2.82 s at twenty-four snapshots, and 11.4 s
against 10.2 s over the year. For a horizon 122 times longer, the nimopt build
time grows 13 times and the linopy build time grows 3.6 times.

**Over the year the nimopt build materialises every expression twice.**
`create_model` declares the constraints, and the shape pass of the
declaration materialises each expression to find its rows and count its
coefficients. `assemble` materialises each expression again to write it. A
constraint with `over=` or `where=` materialises its expression within its
rows. The backend declares the nodal balance once per connectivity group, and
each group materialises its own rows. The backend passes the position of each
entry in each set, and nimopt resolves no label of an entry. nimblend reads
the position of a key from a table over the range of the keys when that range
plus the number of keys is at most four times the number of keys looked up.
It writes the coordinates of a domain with every cell of its shape one range
per dimension.

**The matrix is smaller by a quarter, at every size.** 556 MB against 742 MB
for the same 46.4 million nonzeros. nimopt stores a column index in four bytes
and linopy in eight. The difference is structural and does not depend on the
machine.

**The nimopt solve takes half the time of the linopy solve.** 243.0 s against
505.9 s, for matrices of the same shape passed to the same HiGHS with its
default method. Two more linopy solves, under two other hash seeds and two at
a time, took 536.7 s and 542.7 s. Three nimopt solves, under three other hash
seeds and three at a time, took between 271.4 s and 272.7 s, with 256,469
simplex iterations each. The cause of the difference between the backends is
not identified. The second assembly in the nimopt solve takes less than the
0.89 s of the whole nimopt build. The solve is about 270 times the nimopt
build and about 180 times the linopy build, and it determines the elapsed time
of a run.

**The nimopt backend adds less resident memory in the solve and in the build
of the whole year.** The solve adds 382 MB against 524 MB, and its peak is
135.6 MB lower. Over the year the build adds 3.20 GB against 4.75 GB, and the
process peaks 1.58 GB lower. At twenty-four snapshots the build adds 17.8 MB
against 14.5 MB, and the two processes peak at 819 MB and 957 MB.

## What was not measured

The rungs of 192 and 730 snapshots are not solved here. The whole year is
solved with Mosek on the nimopt backend only.

## Reproducing the numbers

From `nimopt/benchmarks`, the network is downloaded from Zenodo record 15656090
into `data/large/interconnected-transport.nc`, and each horizon is measured:

```bash
curl -L -o data/large/interconnected-transport.nc \
  https://zenodo.org/api/records/15656090/files/RESULTS-EU-interconnected-transport.nc/content
export PYTHONHASHSEED=0
python -c "import bench_vs_linopy as b; print(b.compare('pypsa', {'snapshots': 24}, traced=False, trace_dir=b.TRACES))"
python -c "import bench_vs_linopy as b; print(b.compare('pypsa', {'snapshots': 2920}, traced=False, solve=False, trace_dir=b.TRACES))"
python bench_pypsa_backend.py nimopt --build-only --singleton-rows bounds
python bench_pypsa_backend.py nimopt --build-only --singleton-rows rows
python bench_pypsa_backend.py nimopt --solver mosek --method barrier --crossover off --singleton-rows bounds
python bench_pypsa_backend.py nimopt --solver mosek --method barrier --crossover off --singleton-rows rows
```

PyPSA takes the direction of each Kirchhoff cycle from the iteration order of
a set of bus names. Without a fixed hash seed the coefficients of the
`Kirchhoff-Voltage-Law` rows of some cycles differ in sign between two
processes, on both backends, and no other part of the model differs.

Each run writes the resident size of every phase to
`data/large/traces/pypsa-<backend>-<snapshots>-<phase>.csv`, as
`seconds,rss_mb`.
