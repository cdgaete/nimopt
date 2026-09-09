# What the European network costs each builder

`RESULTS-EU-interconnected-transport.nc`, a sector-coupled European capacity
expansion network: 485 buses, 2920 snapshots, 469 generators, 71 lines, 1911
links, 278 stores, 56 storage units. Both builders state the same matrix at
every horizon measured, agreeing on rows, columns and nonzeros, and on the
objective where the problem was solved.

## Twenty-four hours, solved on both sides

162,582 rows by 79,457 columns, 382,520 nonzeros, 4.81 per column, 2,314 dense
columns. Both reach 232,416,019,336.75.

| | nimopt | linopy |
| --- | --- | --- |
| build | 183.9 ms | 2,001.1 ms |
| matrix | 4.6 MB | 6.1 MB |
| solve | 190.2 s | 198.5 s |
| resident added by the solve | 202.9 MB | 295.7 MB |
| highest the solve reached | 456.6 MB | 756.3 MB |
| highest the process reached | 461.9 MB | 1,065.1 MB |

## The whole year, built on both sides

19,423,878 rows by 9,361,137 columns, 46,377,948 nonzeros, 4.95 per column,
2,391 dense columns.

| | nimopt | linopy |
| --- | --- | --- |
| build | 32.0 s | 21.9 s |
| matrix | 557 MB | 742 MB |

These two were measured one after the other in a single process while a solve
ran beside them, so the times carry contention and the resident figures carry
whatever the first left behind. The matrix sizes do not: they are what each
builder allocates for the same 46,377,948 nonzeros.

## What the numbers say

**The build advantage is a fixed cost, and it amortises.** `linopy` spends
about two seconds before it has built anything, which is most of what it costs
at twenty-four hours and a tenth of what it costs over a year. Against that,
`nimopt` goes from 184 ms to 32 s for a hundred and twenty-two times the
horizon — it grows faster than the problem — while `linopy` goes from 2.0 s to
21.9 s. The eleven-fold advantage at the small horizon is one and a half times
the other way at the full one.

**The matrix is smaller by a quarter, at every size.** 557 MB against 742 MB
for the same 46.4 million nonzeros: four-byte column indices against eight.
This is structural and owes nothing to the machine it was measured on.

**The solve belongs to the solver, not the builder.** 190.2 s against 198.5 s
on an identical matrix handed to the same HiGHS. Whatever separates these two
libraries, it is not what happens after the matrix is built — and at
twenty-four hours the solve is a thousand times the build, so it is also the
only part a modeller waits on.

**Resident memory peaks at the end of the solve.** The trace is flat for
seven-eighths of the run — 403 MB for `nimopt`, 722 MB for `linopy` — and then
steps up in the final eighth to 456 MB and 756 MB. A barrier factorization
holds steady and crossover allocates on top of it. A reading taken at the
start would have missed this entirely.

## What was not measured

The 1,279,950-row horizon did not finish a solve in two hours and eighteen
minutes on one side, and the 4,858,188-row horizon was not attempted. Solve
time grows near the 1.8 power of the row count here, so those cost hours and
report what HiGHS costs rather than what either builder does.

Traces of resident size through each phase are written beside the table as
`seconds,rss_mb`.
