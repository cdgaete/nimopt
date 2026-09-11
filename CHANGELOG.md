# Changelog

Every change a user of `nimopt` can observe is listed here, under the release
that carries it. Work that has landed and is not yet released is listed under
`Unreleased`; a release renames that section to its version and date and opens
a new `Unreleased` above it. `tests/test_changelog.py` holds the latest
released section to the version the package states, so a release without a
section here, or a section without the version bump, fails the suite.

The format is the one at <https://keepachangelog.com/en/1.1.0/>, and the
versions follow <https://semver.org/spec/v2.0.0.html>.

## Unreleased

### Fixed

- A set whose members are `datetime64` or `timedelta64` round trips exactly
  through both save formats, in every unit, with and without a fixed member.
  A conversion that is not exact raises `ValueError` instead of truncating.

### Changed

- The model file format is version 3. A version 2 file still loads. A set of
  `datetime64` or `timedelta64` members is written inline as a mapping of
  `dtype` and `members`.
- A member fixed in a relation is written as text: a `datetime64` member as
  its quoted ISO 8601 string, and a `timedelta64` member as a quoted count
  and numpy unit code such as `'3 h'`.
- `Model.row` takes an ISO 8601 string for a `datetime64` dimension and a
  count and unit code for a `timedelta64` dimension. A `Row` displays each
  such coordinate in the same text.
- A member of a `datetime64` set specifies no time zone. A string with an
  offset or a trailing `Z`, and a `datetime.datetime` with a `tzinfo`, raise
  `ValueError`.
- A member outside the range of its `datetime64` or `timedelta64` dtype raises
  `ValueError` and reports the first and the last member that dtype
  represents. A `NaT` member raises `ValueError` and reports that it is not a
  time.

## 0.2.2 - 2026-09-11

### Changed

- `Model.row` raises `KeyError` with a message that identifies the dimension
  and the constraint for a label that is not a member of its dimension.

## 0.2.1 - 2026-09-11

### Changed

- `Solution.primal`, `Solution.dual`, `Model.row`, `Model.absent` and
  `Assembled.row_of` raise `KeyError` with a message for a name that is not
  declared. The message lists the declared names of that kind.
  `Solution.primal`, `Solution.dual`, `Model.row` and `Model.absent` report
  the kind of a name of the other kind. `primal` and `dual` check the name
  before `status`, and raise `KeyError` for a name declared after the solve.
- `Session.solve` and `Session.diagnose` raise `ValueError` where the model
  declares a variable or a constraint after the session opened.
- `Solution.gap` returns `None` at status `unbounded` and
  `unbounded_or_infeasible`, as `Solution.bound` does. `Solution.objective`
  and `Solution.primal` raise `ValueError` there.
- The package requires `nimblend` 0.2.1. The `nimblend` messages that nimopt
  passes through report the condition, then the action to take.

## 0.2.0 - 2026-09-11

### Added

- `save`, `dumps`, `Model.to_yaml` and `Definition.to_yaml` take
  `instructions=True`. The written file then opens on a fixed comment block
  that explains the format to a reader without the package. The block is a
  YAML comment; a file with the block loads to the same definition as one
  without it.

### Changed

- `Model.constraint` and `Definition.constraint` replace the method `eq`;
  every relation goes through them.
- `Capabilities.rejected` and `Capabilities.rejects(one, other)` replace
  `refused` and `refuses`. The repr writes `rejects a+b`.
- `Param.expresses` and `Variable.expresses` replace `Param.states` and
  `Variable.states`.
- Each solver adapter returns a `Result` from `solve`, with `status`,
  `feasible`, `objective`, `bound`, `col_value`, `row_dual` and `backend`.
  `Result` validates `feasible`, `objective` and `bound` against `status`.
- `Solution.objective` and `Solution.primal` return the point a solver reports
  at a limit. They raise `ValueError` where the solver reports no feasible
  point, and `Solution.dual` raises where `status` is not `optimal`.
- `Solution.objective`, `Solution.primal` and `Solution.gap` raise
  `ValueError` at status `unbounded` and `unbounded_or_infeasible`, whatever
  `Solution.feasible` reports. The three adapters agree at those statuses.
- `Solution.feasible` reports whether the solver found a primal-feasible
  point. `Solution.bound` is the bound on the optimal objective the solver
  proved, or `None`. `Solution.gap` is the relative distance from the
  objective to that bound, or `None`.
- `Solution.__repr__` reports the objective of a feasible point, and the gap
  beside it where the solver proved a bound. It writes `no values` where a
  value read raises.
- The error messages and the docstrings are written in technical English.
  Each message is a condition followed by the action to take. The text of
  several messages changed.
- The package supports Python 3.12, 3.13 and 3.14, and numpy from 2.3, and
  requires `nimblend` 0.2.0.
- The install section of the documentation opens on the PyPI install.

## 0.1.2 - 2026-09-09

### Fixed

- The playground opens on an example that runs. A parameter states no
  coefficient until it is read at its sets, so the opening example reads
  `supply[P]` and `demand[W]`, and a test runs the seed.

## 0.1.1 - 2026-09-09

### Changed

- The README and the description on the package index name the documentation
  site.
- The playground names the wheels this version builds, and the site deploy
  fetches the `nimblend` release beside it.

## 0.1.0 - 2026-09-08

### Added

- An LP/MILP builder in which a variable is a dimension. Sets, parameters,
  variables and constraints are declared symbolically and expanded into a
  matrix at write or solve time, with the constraint blocks carried as
  `nimblend` labeled sparse arrays.
- `Definition` for a model declared before its data exists, and `Model` for
  one bound to data, with `absent()` and `row()` for inspecting the rows the
  matrix carries and the rows it drops.
- Solver adapters for HiGHS, Gurobi and Mosek, with `available()`,
  `capabilities()` and `options()` for what each adapter takes, and `Session`
  for solving against a live backend.
- A YAML model file with an `.npz` sidecar or inline data, written by `save`
  and read by `load` and `loads`.
- The documentation site at <https://cdgaete.github.io/nimopt/>, with a
  playground that runs every example in the browser.
