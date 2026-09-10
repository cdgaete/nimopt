# What the European network costs each builder

`RESULTS-EU-interconnected-transport.nc`, a sector-coupled European capacity
expansion network: 485 buses, 2920 snapshots, 469 generators, 71 lines, 1911
links, 278 stores, 56 storage units. Both builders produce the same matrix at
every horizon measured. They agree on rows, columns and nonzeros, and on the
objective where the problem was solved.

## Twenty-four hours, solved on both sides

162,582 rows by 79,457 columns, 382,520 nonzeros, 4.81 per column, 2,314 dense
columns. Both objectives are 232,416,019,336.75.

| | nimopt | linopy |
| --- | --- | --- |
| build | 183.9 ms | 2,001.1 ms |
| matrix | 4.6 MB | 6.1 MB |
| solve | 190.2 s | 198.5 s |
| resident added by the solve | 202.9 MB | 295.7 MB |
| peak resident of the solve | 456.6 MB | 756.3 MB |
| peak resident of the process | 461.9 MB | 1,065.1 MB |

## The whole year, built on both sides

19,423,878 rows by 9,361,137 columns, 46,377,948 nonzeros, 4.95 per column,
2,391 dense columns.

| | nimopt | linopy |
| --- | --- | --- |
| build | 32.0 s | 21.9 s |
| matrix | 557 MB | 742 MB |

Both runs are measured in one process, and a solve runs on the same machine.
The times include that contention and the resident figures include the memory
of the first run. The matrix sizes do not: each is what the builder allocates for
the same 46,377,948 nonzeros.

## What the numbers show

**The build advantage is a fixed cost, and it amortizes.** `linopy` spends
about two seconds before the first row is built. That is most of its cost at
twenty-four hours and a tenth of its cost over a year. `nimopt` goes from
184 ms to 32.0 s for a horizon 122 times longer. The build time grows faster
than the problem. `linopy` goes from 2.0 s to 21.9 s. The eleven-fold nimopt
advantage at the small horizon is a 1.5-fold linopy advantage at the full one.

**The matrix is smaller by a quarter, at every size.** 557 MB against 742 MB
for the same 46.4 million nonzeros: four-byte column indices against eight-byte
ones. The difference is structural and does not depend on the machine.

**The solve time is a property of the solver.** 190.2 s against 198.5 s on an
identical matrix passed to the same HiGHS. At twenty-four hours the solve is a
thousand times the build, and the solve dominates the elapsed time of a run.

**Resident memory peaks at the end of the solve.** The trace is flat for
seven-eighths of the run, at 403 MB for `nimopt` and 722 MB for `linopy`. It
rises in the final eighth to 456 MB and 756 MB. A barrier factorization
allocates a constant amount and crossover allocates on top of it. A single
reading at the start does not record the peak.

## What was not measured

The 1,279,950-row horizon did not finish a solve in two hours and eighteen
minutes on one side. The 4,858,188-row horizon was not attempted. Solve time
grows near the 1.8 power of the row count here. Those horizons cost hours and
measure HiGHS and not either builder.

Traces of resident size through each phase are written beside the table as
`seconds,rss_mb`.
