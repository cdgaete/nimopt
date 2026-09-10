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

### Added

- `save`, `dumps`, `Model.to_yaml` and `Definition.to_yaml` take
  `instructions=True`, which opens the written file on a fixed comment block
  explaining the format, so that a reader given one file alone can interpret
  it without the package. The block is a YAML comment, so a file carrying it
  reads to the same definition as one without it.

### Changed

- `Solution.objective` and `Solution.primal` return the point a solver reports
  at a limit. They raise `ValueError` where the solver reports no feasible
  point.
- `Solution.feasible` reports whether the solver found a primal-feasible
  point. `Solution.bound` is the bound on the optimal objective the solver
  proved, or `None`. `Solution.gap` is the relative distance from the
  objective to that bound, or `None`.
- `Solution.dual` raises `ValueError` where `status` is not `optimal`.
- `Solution.__repr__` reports the objective of a feasible point at any status,
  and the gap beside it where the solver proved a bound.
- Each solver adapter returns a `Result` from `solve`, with `status`,
  `feasible`, `objective`, `bound`, `col_value`, `row_dual` and `backend`.
  `Result` validates `feasible`, `objective` and `bound` against `status`.
- The package supports Python 3.12, 3.13 and 3.14, and numpy from 2.3.
- `Model.constraint` and `Definition.constraint` replace the method `eq`;
  every relation goes through them.
- The install section of the documentation opens on the PyPI install.
- `Capabilities.rejected` and `Capabilities.rejects(one, other)` replace
  `refused` and `refuses`. The repr writes `rejects a+b`.
- The error messages and the docstrings of the model, file and solver layers
  are written in technical English. Each message is a condition followed by
  the action to take. The text of several messages changed.

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
