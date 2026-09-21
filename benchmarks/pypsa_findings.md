# What the European network costs each PyPSA backend

`RESULTS-EU-interconnected-transport.nc`, a sector-coupled European capacity
expansion network: 485 buses, 2920 snapshots, 469 generators, 71 lines, 1911
links, 278 stores, 56 storage units. PyPSA builds its model through
`n.optimize.create_model` on the `nimopt` backend and on the `linopy` backend.
The `pypsa` rung of `bench_vs_linopy.py` measures both, each in a process of
its own. At both horizons measured, the two backends build matrices with the
same rows, columns and nonzeros. Where the problem was solved, both solve to
the same objective.

## Setup

An AMD Ryzen 7 5800X with 8 cores, 16 threads and 62 GB of memory. highspy
1.15.1, PyPSA 1.3.0, linopy 0.9.1 and numpy 2.5.2. nimopt at `5f3d358` and
nimblend at `733ed1c`. The one-minute load average read 1.1 before the first
rung and 4.2 after each rung.

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

162,582 rows by 79,457 columns, 382,520 nonzeros, 4.81 per column, 2,314 dense
columns. Both objectives are 232,416,019,336.75.

| | nimopt | linopy |
| --- | --- | --- |
| build | 1,138.6 ms | 2,159.2 ms |
| matrix | 4.6 MB | 6.1 MB |
| solve | 285.4 s | 210.7 s |
| read | 4.6 s | 5.2 s |
| resident added by the solve | 368.0 MB | 484.3 MB |
| peak resident of the solve | 829.2 MB | 947.2 MB |
| peak resident of the process | 1,065.8 MB | 1,065.9 MB |

The generator dispatch sums to 118,174,790.05 on both backends. The nodal
prices sum to 10,632,436.32 on the nimopt backend and to 10,636,519.87 on the
linopy backend. The two solves return different dual points.

## The whole year, built on both backends

19,423,878 rows by 9,361,137 columns, 46,377,948 nonzeros, 4.95 per column,
2,391 dense columns.

| | nimopt | linopy |
| --- | --- | --- |
| build | 69.5 s | 18.1 s |
| matrix | 557 MB | 742 MB |
| resident added by the build | 4,321 MB | 4,769 MB |
| peak resident of the process | 5,035 MB | 5,513 MB |

A separate nimopt run times the two parts of the build: `create_model` takes
37.5 s and `assemble` takes 35.3 s.

## What the numbers show

**The nimopt build is faster at twenty-four snapshots and slower over the
year.** It takes 1.14 s against 2.16 s at twenty-four snapshots, and 69.5 s
against 18.1 s over the year. For a horizon 122 times longer, the nimopt build
time grows 61 times and the linopy build time grows 8.4 times.

**Over the year the nimopt build has two parts of equal size.**
`create_model` translates the network and declares the nimopt model in 37.5 s.
`assemble` writes the matrix in 35.3 s. linopy builds its model and its matrix
in 18.1 s.

**The matrix is smaller by a quarter, at every size.** 557 MB against 742 MB
for the same 46.4 million nonzeros. nimopt stores a column index in four bytes
and linopy in eight. The difference is structural and does not depend on the
machine.

**The nimopt solve takes 35% longer.** 285.4 s against 210.7 s, for matrices
of the same shape passed to the same HiGHS with its default method. The second
assembly in the nimopt solve takes less than the 1.14 s of the whole nimopt
build. One run per side does not identify the cause of the difference. The
solve is about 250 times the nimopt build and about 98 times the linopy build,
and it determines the elapsed time of a run.

**The nimopt backend adds less resident memory in the solve and in the build
of the whole year.** The solve adds 368 MB against 484 MB, and its peak is
118 MB lower. Over the year the build adds 4.3 GB against 4.8 GB. At
twenty-four snapshots the build adds 16.1 MB against 13.6 MB, and the two
processes peak at the same 1,066 MB.

## What was not measured

The rungs of 192 and 730 snapshots are not solved here, and the whole year is
built and not solved.

## Reproducing the numbers

From `nimopt/benchmarks`, the network is downloaded from Zenodo record 15656090
into `data/large/interconnected-transport.nc`, and each horizon is measured:

```bash
curl -L -o data/large/interconnected-transport.nc \
  https://zenodo.org/api/records/15656090/files/RESULTS-EU-interconnected-transport.nc/content
python -c "import bench_vs_linopy as b; print(b.compare('pypsa', {'snapshots': 24}, traced=False, trace_dir=b.TRACES))"
python -c "import bench_vs_linopy as b; print(b.compare('pypsa', {'snapshots': 2920}, traced=False, solve=False, trace_dir=b.TRACES))"
```

Each run writes the resident size of every phase to
`data/large/traces/pypsa-<backend>-<snapshots>-<phase>.csv`, as
`seconds,rss_mb`.
