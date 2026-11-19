# Changelog

Every change a user of `nimopt` can observe is listed here, under the release
that carries it. Work that has landed and is not yet released is listed under
`Unreleased`; a release renames that section to its version and date and opens
a new `Unreleased` above it. `tests/test_changelog.py` holds the latest
released section to the version the package states, so a release without a
section here, or a section without the version bump, fails the suite.

The format is the one at <https://keepachangelog.com/en/1.1.0/>.

A version is `MAJOR.YYYYMMDD.PATCH`. `MAJOR` is 0 for a package that is not
stable, and any release of it can contain a breaking change. `MAJOR` is 1 for
the first stable release and increases with every breaking change. `YYYYMMDD`
is the date of a release that adds features. `PATCH` counts the releases that
only correct defects of that release, from 0. The releases up to 0.4.1 follow
<https://semver.org/spec/v2.0.0.html>.

## Unreleased

### Added

- `singleton_rows` on `Model.assemble`, `Model.session` and `Model.solve`, with the values `"bounds"` and `"rows"`.
- `explain()` reports the rows with one coefficient of each constraint and of the model.
- On HiGHS, `Diagnosis.ray` contains a ray for an unbounded model with no row. HiGHS returns none for such a model, and the adapter computes it from the costs and the bounds of the columns.

### Changed

- A row with one coefficient is stored as a bound of its column, and the solver receives a matrix without it. `singleton_rows="rows"` on `Model.assemble`, `Model.session` and `Model.solve` builds the matrix with every row.
- `Row.index` is the position of the row among the declared rows.
- `Session.n_rows` and `Assembled.n_rows` report the rows the solver loaded.
- `Solution.dual` returns a value at every declared row. The dual of a row stored as a column bound is computed from the reduced costs.
- The `conflict` function of the HiGHS and Gurobi adapters returns a third array: for each column, which of its bounds is in the subsystem. `diagnose()` reports a row stored as a column bound where its bound is in the conflict, and reports no `ColumnBound` for a column whose bounds in the conflict are all set by such rows.
- On HiGHS, `diagnose()` raises `RuntimeError` for a subsystem with no row and no column, as it does for an invalid subsystem.

## 0.20261118.1 - 2026-11-18

### Fixed

- A variable, a constraint or an objective that reads two sets of one name with different labels raises `ValueError` where it is declared, and the message identifies it. For a variable the check covers a parameter given as `lower=` or `upper=`, to `Variable.align`, to `Session.fix` or to `Session.set_bound`, and its `subset=`. For a constraint and for the objective it covers every variable, coefficient, condition and right-hand side, with `over=` and the `where=` of `Sum`. A set read at one fixed member is compared as well: `p["t0"] * x[A, "t0"]` raises where `p` and `x` are over two sets named `T` with different labels.
- Two sets of one name with equal labels in the same order are one dimension. `Model.explain()` and `save` report such a set once. They raise `ValueError` for two sets of one name with different labels.

## 0.20261118.0 - 2026-11-18

### Added

- `Expression.block()` returns the coefficients of an expression as a nimblend array and computes no row domain. `Expression.to_dense()` returns them as an ndarray over the frame and the column dimension, with each term added into one array.
- `Param.domain()` returns the coordinates at which a parameter has a value. Callers that keep the domain of one bound array share one object.
- `Solution.iterations` returns the simplex iterations the solver reports for the solve. The HiGHS adapter reports the count. The value is `None` for the Gurobi, Mosek and GAMS adapters.
- The capability `mutation` reports whether an adapter changes the model a solver has loaded. The HiGHS adapter declares it `native` and defines `change_column_bounds`, `append_rows` and `resume`: a change of column bounds, an append of rows and a solve of the loaded model. The Gurobi, Mosek and GAMS adapters declare it `absent`.
- `Session.fix(target, value)`, `Session.unfix(target)` and `Session.set_bound(target, lower=None, upper=None)` change the column bounds of the model the solver has loaded, and `Session.add_row(name, relation, where=None, over=None)` appends rows to it. `Session.solve()` then solves the loaded model again. With the simplex method HiGHS starts from the basis of the previous solve. `Session.n_rows` and `Session.n_columns` return the size of the loaded model. The model the session was opened on is not changed. The four methods require an adapter with the capability `mutation`, and raise `ValueError` on another adapter.

### Changed

- The HiGHS adapter passes the assembled arrays to HiGHS and builds no `HighsLp` object. For a model of 19.4 million rows, 9.4 million columns and 46.4 million nonzeros the pass takes 1.2 s instead of 7.6 s, and the process holds 1.6 GB instead of 2.7 GB for the model during the solve. The adapter raises `RuntimeError` for an assembled vector whose length differs from the matrix, and the message gives each such vector with both lengths. It raises `RuntimeError` for an integrality flag other than 0 and 1.
- A term summed over several sets reduces them in one pass. The values at one column are added in the order of their coordinates, and the last bits of such a coefficient can differ from the sum taken one set at a time.
- `Model.objective_coefficients` adds each term of the objective into one vector and merges no term blocks.
- A constraint computes no row domain from its terms where it is written into the matrix, and where its rows are declared with `over=`.
- A constraint whose right-hand side is a number stores one float for it, and no value per row. Constraints declared with `over=` one parameter share one row domain.
- Declaring a variable takes the same time whatever the number of variables the model declares. A model declares 32,000 variables in 0.11 s instead of 7.0 s.
- With nimblend's one-pass reductions and its product over shared dimensions, PyPSA's European network over 2,920 snapshots builds in 11.2 s of processor time instead of 14.0 s. The traced peak of the build is 3,085 MB instead of 3,348 MB, and the declared model holds 1,263 MB instead of 1,440 MB. The assembled matrix is identical.

### Fixed

- An objective that is a constant alone gives every column a cost of zero, and the solved objective is the constant.

## 0.20261023.1 - 2026-10-23

### Fixed

- The `gams` adapter passes `crossover` to CPLEX as `solutiontype`: `choose` is 0, `off` is 2 and `on` is 1. CPLEX 22 replaces `barcrossalg -1` with `solutiontype 2` and reports the replacement in its log.
- Where CPLEX terminates without reaching a limit the options set, the `gams` adapter raises a `RuntimeError` that states it and points to the CPLEX log through `log=True`.

## 0.20261023.0 - 2026-10-23

### Added

- The `gams` adapter solves with CPLEX through the GMO and GEV libraries of a GAMS installation, loaded with `ctypes`, on Linux, macOS and Windows. It passes the matrix in memory and requires no GAMS Python package. It requires the `gams` executable on `PATH` and a GAMS license that includes CPLEX. It computes no conflict and no ray.
- The option `separate_process` runs the solver in a process of its own. The `gams` adapter supports it; the other adapters raise `ValueError` for it.

### Changed

- Every solver adapter defines `installed()`. `available()` lists the adapters whose `installed()` returns True.

## 0.20260923.0 - 2026-09-23

### Added

- `Param.from_array` builds a parameter from a nimblend array whose labels are
  members of the sets, in any order and extent, including a solved model's
  primals and duals.
- `Param.from_positions(name, sets, index, values)` returns a parameter from
  an index matrix with one row per set and one value column. It resolves no
  labels.
- `Definition.build` accepts a nimblend array or a `Param` as the data of a
  parameter.
- `tests/test_pdlp.py` solves with `options={"method": "pdlp"}` and checks
  the optimum, the row duals against simplex and the PDLP line of the HiGHS
  log. It is skipped unless `NIMOPT_TEST_PDLP=1`. PDLP on a GPU requires an
  NVIDIA driver and a HiGHS built from source with CUDA.

### Changed

- `Param(name, sets, array)` raises `ValueError` at construction for an array
  whose labels differ from its sets' labels.
- A constraint with `over=` or `where=` materialises its expression within
  those rows. Each term restricts its variable's columns and its coefficient
  to the rows before the product. `Model.absent` materialises the whole
  expression and reports every narrowing.
- The assembly looks up the rows of each constraint once. The number of
  coefficients is checked against the number measured at declaration before
  any entry is written.
- `Expression.materialise` and `Expression.materialise_at` add the term
  blocks with `nimblend.sum_arrays`, in one merge. The values are added in
  the order of the terms, and the block is equal to the sum computed with `+`.
- A `Variable` numbers its own coordinate from 0 and passes no `start` to a
  `nimblend` coordinate.
- The `pypsa` rung of `benchmarks/bench_vs_linopy.py` builds the European
  network through the nimopt and the linopy backend of PyPSA.
  `benchmarks/bench_pypsa_backend.py` reports the linopy column count without
  the objective constant column.
- `pypsa_network.py`, `pypsa_fidelity.py` and `pypsa_reference.py` are in
  `tests/`. `pypsa_network.py` is the PyPSA-free fixture of
  `tests/test_pypsa_fidelity.py`.
- The `nimblend` domains page documents `as_coord()` and the coordinates
  without a `start`.
- `benchmarks/pypsa_findings.md` reports the European network built by PyPSA
  on its nimopt and its linopy backend: solved at 24 snapshots, and built over
  the 2920 snapshots of the year.

### Removed

- `benchmarks/pdlp_gpu.py`, which passed the matrix to highspy outside
  `Session`, and `benchmarks/bench_pypsa.py`, which built a second
  formulation of the PyPSA model.

## 0.20260921.0 - 2026-09-21

### Added

- `Constraint.labels_at(positions)` and `Variable.labels_at(positions)`
  return the labels of the rows or the columns at those positions.
- `Constraint.position_of(coords)` returns the position of the row at a
  coordinate.
- `dumps(what, inline=False, instructions=False, version=4)` returns the text
  of a definition's file or a model's file. `loads` reads it.
- `Constraint.row_at(position)` returns the columns, the coefficients and the
  bounds of one row. `Expression.materialise_at(at)` returns the coefficients
  at one coordinate of the frame.

### Changed

- `Model.row` and `Session.diagnose` read the labels of the rows and the
  columns they report alone. They decode no other row of the constraint.

### Removed

- `Model.to_yaml` and `Definition.to_yaml`. `dumps` returns the same text.

### Fixed

- `Session.solve` and `Session.diagnose` raise `ValueError` where the model
  sets its objective after the session opened.
- `Model.row` computes the entries of the named row alone. It builds no
  matrix and no other row.
- `Model.row(name)` returns the one row of a constraint with no free
  dimension.

## 0.20260919.0 - 2026-09-19

### Changed

- `Model.row` raises `ValueError` for a coordinate label that is not a member
  of the constraint's dimension.
- The message for a `Sum(..., where=)` condition with no name matches the
  message `files.structure` raises for the same defect: it identifies the
  owner and the slot.
- A coefficient, a parameter and a variable read at dimensions other than
  their own raise one wording, `is declared over`.
- `Constraint.write_into` takes the coordinate of every row of the model, and
  numbers the constraint's rows inside it from `row_start`. It calls the
  nimblend `group(coord=, start=)`: nimopt requires the nimblend release that
  contains it.
- A primal or a reduced cost of a variable whose `subset=` covers the product
  of its sets is a `DenseArray`. A primal and a dual follow one rule: a
  `DenseArray` where the members cover the product, and a `SparseArray`
  otherwise.

### Added

- `product` takes domains beside sets. A domain contributes its own members,
  and `product((subset((G,), columns), T))` crosses a subset with a set. Two
  factors over one dimension raise `ValueError`.
- `Variable.bound_array(which)` returns the lower or the upper bound at each
  member of the variable: a float, or an array over the variable's
  dimensions. The column bounds and the checks of `Model.piecewise` read it.
- `Model.piecewise` and `Definition.piecewise` take `method="auto"`. It
  generates `"tangent"` where the sign is not `"=="`, `active` is None, `x`
  has no constant and the curvature of every entity matches the sign, and
  `"incremental"` otherwise. `Piecewise.formulation` reports the method a
  model generates. A definition reserves the generated names of both methods.
  A model file writes `method: auto`.
- `Solution.has_duals` returns True where `Solution.dual` returns values:
  at status `optimal`, for a solve whose solver reports duals.
- `Piecewise.entity` returns the dimensions of `x_points` other than the
  breakpoint set, and `Piecewise.where_domain()` returns the domain of
  `where` over them. `Piecewise.check_active` takes that domain as `where`.

### Fixed

- A single `Set`, `Alias`, lagged set or cyclic set given to `Variable`,
  `Param`, `Param.from_dense`, `Param.from_long`, `product`, `subset` or
  `subset_of` in place of a list of sets raises `TypeError` with the owner,
  the class of the object and a hint.
- A bound message writes a timedelta member as a count and a unit, such as
  `'3 h'`, and `Diagnosis` writes a datetime coordinate as its ISO text. Every
  message and display writes a member in the form a model file reads.
- `Model.piecewise` reads the bounds of an `active` variable at the members
  of the variable. A bound parameter above 1 outside the variable's `subset=`
  raises no `ValueError`.
- `where=`, `over=` and `subset=` raise `ValueError` for a tuple that
  contains a lagged or a cyclic set, such as `(T - 1,)`, in
  `Model.constraint`, `Model.var`, `Sum` and the `Definition` methods. A
  `Definition` raises when the declaration is made.
- A condition that is not a parameter, a tuple of sets or a domain, such as a
  list or a string, raises `ValueError`.
- A set, an alias, a parameter, a variable or a piecewise declaration named by
  a Python keyword, such as `lambda`, raises `ValueError`. A `Definition`
  raises when the declaration is made, and a `Model` when it is written.
- Writing a `Model` raises `ValueError` for one name given to two kinds of
  symbol, such as a variable and a parameter. `loads` raises for such a file.

## 0.4.1 - 2026-09-18

### Fixed

- `Piecewise.check_active` reads the bounds of an `active` variable at the
  coordinates of `where`. An upper bound above 1 outside `where`, such as the
  status of a modular unit that counts modules, raises no `ValueError`.

## 0.4.0 - 2026-09-18

### Added

- `Model.piecewise` and `Definition.piecewise` take `where=`: a parameter, a
  tuple of sets or a domain over the sets of `x_points` other than the
  breakpoint set. The breakpoint checks, the generated columns and the
  generated rows cover the entities at its coordinates. `x`, `y` and
  `active` are compared at those coordinates only. Two declarations can
  share breakpoints and split the entities.
- A model file writes the key `where` under a piecewise declaration that sets
  it: a parameter's name or a list of set names. The format version is 4.

## 0.3.1 - 2026-09-17

### Fixed

- A timedelta member written with a unit multiplier of 0, such as `'2 0h'`,
  raises `ValueError`. numpy 2.5.2 ends the Python process with a
  floating-point exception when it converts such a unit.

### Changed

- A timedelta member written with the unit `generic`, and an integer member
  of a timedelta set whose dtype has the generic unit, raise `ValueError`.
  numpy deprecates the generic unit.

## 0.3.0 - 2026-09-17

### Added

- `Model.piecewise` and `Definition.piecewise` declare a piecewise-linear
  relation of one expression to another through breakpoints. `x` is on the
  curve through `x_points` and `y_points`, and `sign` compares `y` with the
  curve. The declaration generates the sets, parameters, variables and
  constraints of its method under names that begin with its own.
- The `incremental` method generates one continuous and one integer variable
  per segment, and is exact for breakpoints that are strictly increasing or
  strictly decreasing. The `tangent` method generates one row per segment for
  points that are convex under `>=` or concave under `<=`.
- `active=` takes a binary variable, or a sum of them, over the sets of `x`.
  Where it is 0, `x` is 0 and `y` is compared with 0. The `incremental`
  method supports it. A term that is scaled or bounded outside 0 and 1 raises
  `ValueError`, and so does a continuous term under the default.
- `Model.piecewise` and `Definition.piecewise` take `relaxed=`. `True`
  accepts a continuous `active` between 0 and 1 and declares the scaled
  curve, the linear relaxation of the switch. A value between 0 and 1 scales
  every breakpoint of the curve. `relaxed=True` with no `active=` raises
  `ValueError`. A model file writes the key `relaxed` under a declaration
  that sets it.
- A piecewise declaration whose `x`, `y` and `active` are over different
  members raises `ValueError` and reports the first member they differ at.
  A curve relates one column of `x` to one column of `y`, and a member that
  one of them does not have relates a column to nothing.
- A variable or a constraint that takes a name a piecewise declaration
  generates raises `ValueError`, in a model and in a definition. A file
  writes a set, a parameter and a variable into one table of symbols, and one
  name for two of them is a file that does not load.
- A piecewise declaration allocates nothing for a member of its sets that has
  no breakpoint. The breakpoints are read as arrays over the members that
  have one.
- A model generates the declarations when `piecewise` is called. A definition
  stores the declaration, and `build` generates it once the data is bound.
- `Piecewise` is exported. `Model.piecewise_declarations` and
  `Definition.piecewise_declarations` contain each declaration by name.
  `Explanation.piecewise` reports each one with the names it generated.
- `save`, `Model.to_yaml` and `Definition.to_yaml` take `version=4` or
  `version=3`.
- `Solution.dual` takes a variable and returns its reduced costs over the
  variable's own sets: the objective coefficient less the duals of the rows
  the variable appears in, weighted by its coefficients in them. nimopt
  derives the value, so the convention does not vary by solver.
- `Solution.dual` takes `kind="constraint"` or `kind="variable"`. A model
  declares its constraints and its variables in two registries, and a name
  that identifies one of each raises `ValueError` with no `kind`. The message
  for an unknown name lists each declared name once.

### Fixed

- An expression builds its rows where a coefficient introduces a set that
  another term is not over, including where the coefficient and the variable
  have no set in common.
- A coefficient multiplying an expression with a constant raises `ValueError`.
  Such a product is one value per row, and an expression has one constant for
  every row.
- A set member written as `NaT` raises `ValueError` with no numpy
  `DeprecationWarning` before it.
- The message for an unknown key in a model file writes the unknown keys and
  the accepted keys as quoted names separated by commas.

### Changed

- The model file format is version 4. A piecewise declaration is written under
  the key `piecewise`, and the sets, parameters, variables and constraints it
  generated are not written. Loading the file generates them again. Files of
  version 2 and 3 load. `version=3` writes the generated declarations in place
  of the piecewise declaration.
- The package requires numpy 2.5. From that version numpy raises
  `OverflowError` for an overflowing datetime64 unit conversion. A set member
  outside the range of its dimension's dtype reports that range on every
  supported numpy.

## 0.2.3 - 2026-09-11

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
