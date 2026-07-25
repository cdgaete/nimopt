"""Solution extraction and storage."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .model import Model
    from .solvers.base import Solver


@dataclass
class VariableSolution:
    """Solution data for a single variable."""

    name: str
    dims: list[str]
    elements: list[list]  # Elements for each dimension
    values: np.ndarray  # Shape matches variable shape
    duals: np.ndarray  # Reduced costs, same shape

    def to_array(self):
        """Convert values to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.values.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.values, dims=self.dims, coords=coords)

    def to_dual_array(self):
        """Convert reduced costs to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.duals.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.duals, dims=self.dims, coords=coords)


@dataclass
class ConstraintSolution:
    """Solution data for a single constraint."""

    name: str
    dims: list[str]
    elements: list[list]  # Elements for each dimension
    duals: np.ndarray  # Shadow prices

    def to_array(self):
        """Convert duals (shadow prices) to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.duals.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.duals, dims=self.dims, coords=coords)


@dataclass
class Solution:
    """Complete solution with per-variable and per-constraint data."""

    objective_value: Optional[float] = None
    variables: dict[str, VariableSolution] = field(default_factory=dict)
    constraints: dict[str, ConstraintSolution] = field(default_factory=dict)

    def var(self, name: str):
        """Get variable solution values as nimblend Array."""
        return self.variables[name].to_array()

    def var_dual(self, name: str):
        """Get variable reduced costs as nimblend Array."""
        return self.variables[name].to_dual_array()

    def con(self, name: str):
        """Get constraint duals (shadow prices) as nimblend Array."""
        return self.constraints[name].to_array()


def extract_solution_python(
    solver: "Solver", model: "Model", align_by_names: bool = False
) -> Solution:
    """Extract solution from solver using model metadata.

    By default assumes the solver's column/row order matches the model's
    insertion order (true for the direct HiGHS interface, which builds
    the matrices itself).

    align_by_names=True must be used for LP-file-loaded models: the LP
    format has no column declaration, so the solver assigns column
    indices by first textual appearance (objective first) - variables
    with zero objective coefficient end up wherever a constraint first
    mentions them. In that mode the solver's column/row names are
    aligned to the model's expected names with a vectorized permutation
    before slicing; if names are unavailable or ambiguous, positional
    order is kept.
    """
    sol = Solution()

    # Get objective value if available
    if hasattr(solver, '_h'):
        # HiGHS direct solver
        sol.objective_value = solver._h.getInfo().objective_function_value
    elif hasattr(solver, 'get_objective_value'):
        sol.objective_value = solver.get_objective_value()

    # Get flat arrays from solver
    var_values = np.array(solver.get_variable_values(), dtype=np.float64)
    var_duals = np.array(solver.get_variable_duals(), dtype=np.float64)
    con_duals = np.array(solver.get_constraint_duals(), dtype=np.float64)

    # Actual rows present per constraint (lagged constraints emit fewer rows
    # than product(free sizes)); drives both the row permutation and the
    # per-constraint dual slicing below so they stay consistent.
    con_rows = _present_con_rows(model, solver)

    if align_by_names:
        col_perm = _name_permutation(
            _expected_col_names(model),
            solver.get_variable_names(),
            var_values.size,
        )
        if con_rows is not None:
            present_names = [nm for _, _, _, nms in con_rows for nm in nms]
            expected_row_names = (
                np.asarray(present_names, dtype=object) if present_names else None
            )
        else:
            expected_row_names = _expected_row_names(model)
        row_perm = _name_permutation(
            expected_row_names,
            solver.get_constraint_names(),
            con_duals.size,
        )
        if col_perm is not None:
            var_values = var_values[col_perm]
            var_duals = var_duals[col_perm]
        if row_perm is not None:
            con_duals = con_duals[row_perm]

    # Variables: use model metadata to map directly
    var_offset = 0
    for var_name, var in model.variables.items():
        size = var.size
        dims = var.dims
        elements = [list(s.elements) for s in var.sets] if var.sets else []
        shape = var.shape if var.sets else (1,)

        # Slice values in LP order (same as itertools.product)
        values = var_values[var_offset : var_offset + size].reshape(shape)
        duals = var_duals[var_offset : var_offset + size].reshape(shape)
        var_offset += size

        sol.variables[var_name] = VariableSolution(
            name=var_name,
            dims=dims,
            elements=elements,
            values=values,
            duals=duals,
        )

    # Constraints: use model metadata
    present_by_con = (
        {cn: combos for cn, _c, combos, _nms in con_rows} if con_rows is not None else None
    )
    con_offset = 0
    for con_name, constraint in model._constraints.items():
        free_sets = constraint.free_sets
        full_size = 1
        for s in free_sets:
            full_size *= len(s)

        # Number of rows actually emitted for this constraint. Equals full_size
        # for ordinary constraints; smaller for lagged ones with dropped rows.
        combos = present_by_con.get(con_name) if present_by_con is not None else None
        actual = len(combos) if combos is not None else full_size

        if not free_sets:
            shape = (1,)
            dims = []
            elements = []
        elif actual == full_size:
            shape = tuple(len(s) for s in free_sets)
            dims = [s.name for s in free_sets]
            elements = [list(s.elements) for s in free_sets]
        else:
            # Partial (lagged) constraint: report the present rows flat, with the
            # actual per-row index combos rather than a full N-D grid.
            shape = (actual,)
            dims = [s.name for s in free_sets]
            elements = [list(c) for c in combos]

        duals = con_duals[con_offset : con_offset + actual].reshape(shape)
        con_offset += actual

        sol.constraints[con_name] = ConstraintSolution(
            name=con_name,
            dims=dims,
            elements=elements,
            duals=duals,
        )

    return sol


def _expected_col_names(model: "Model"):
    """Expected LP column names (numpy str array) in variable insertion
    order, built vectorized with np.char. Matches
    Variable.all_names(sanitize=True)."""
    from .writers import sanitize_lp_name

    blocks = []
    for var in model.variables.values():
        if not var.sets:
            blocks.append(np.array([var.name]))
            continue
        names = np.array([var.name], dtype=object)
        for s in var.sets:
            elems = np.array([sanitize_lp_name(e) for e in s.elements], dtype=object)
            names = np.char.add(
                np.char.add(names[:, None].astype(str), "_"),
                elems[None, :].astype(str),
            ).ravel()
        blocks.append(names)
    return np.concatenate(blocks) if blocks else None


def _expected_row_names(model: "Model"):
    """Expected LP row names (numpy str array) in constraint insertion
    order, matching the Rust LP writer:
    ``{name}_{elem1}_{elem2}...`` over the free sets in C-order
    (elements sanitized), or the bare name for scalar constraints."""
    from .writers import sanitize_lp_name

    blocks = []
    for con_name, con in model._constraints.items():
        base = sanitize_lp_name(con_name)
        if not con.free_sets:
            blocks.append(np.array([base]))
            continue
        names = np.array([base], dtype=object)
        for s in con.free_sets:
            elems = np.array([sanitize_lp_name(e) for e in s.elements], dtype=object)
            names = np.char.add(
                np.char.add(names[:, None].astype(str), "_"),
                elems[None, :].astype(str),
            ).ravel()
        blocks.append(names)
    return np.concatenate(blocks) if blocks else None


def _model_has_lag(model) -> bool:
    """True if any constraint references a lagged/lead index.

    Only lagged constraints can drop rows, so this gates the (per-row, O(rows))
    presence scan in `_present_con_rows`. A lag can sit in an LHS term, an RHS
    LinearExpr term, or a bare RHS VarRef's `lagged_indices`."""

    def _terms_lagged(expr) -> bool:
        terms = getattr(expr, "terms", None)
        return bool(terms) and any(len(t) > 3 and t[3] for t in terms)

    for con in model._constraints.values():
        if _terms_lagged(con.lhs):
            return True
        rhs = con.rhs
        if _terms_lagged(rhs) or getattr(rhs, "lagged_indices", None):
            return True
    return False


def _present_con_rows(model, solver):
    """Per constraint, the free-index combos whose row is actually present.

    A lagged constraint drops out-of-range rows (e.g. `soc[H-1]` has no row for
    the first H), so it generates fewer than product(free sizes) rows. Returns
    a list of (con_name, con, combos, names) in model order, restricted to rows
    that appear in solver.get_constraint_names(). Returns None if the solver
    exposes no names (caller keeps the naive product-sized behaviour)."""
    import itertools

    from .writers import sanitize_lp_name

    # Fast path: with no lagged constraints every constraint emits its full
    # product(free sizes) rows, so the naive slicing in the caller is exact and
    # correct. Skip the O(total rows) name-building scan entirely -- it used to
    # run on every solve and dominated solution extraction for large models.
    if not _model_has_lag(model):
        return None

    try:
        solver_names = set(solver.get_constraint_names())
    except Exception:
        return None
    if not solver_names:
        return None

    blocks = []
    for con_name, con in model._constraints.items():
        base = sanitize_lp_name(con_name)
        if not con.free_sets:
            combos, names = [()], [base]
        else:
            combos, names = [], []
            for combo in itertools.product(*(s.elements for s in con.free_sets)):
                nm = base
                for e in combo:
                    nm = nm + "_" + sanitize_lp_name(e)
                combos.append(combo)
                names.append(nm)
        keep = [(c, nm) for c, nm in zip(combos, names) if nm in solver_names]
        blocks.append((con_name, con, [c for c, _ in keep], [nm for _, nm in keep]))
    return blocks


def _name_permutation(expected_names, solver_names, n: int):
    """Index array perm with perm[model_flat_index] = solver_index.

    Vectorized via argsort alignment. Returns None (caller keeps
    positional order) if names are unavailable, sizes disagree, names
    don't match, or names are duplicated (mapping would be ambiguous).
    Fast-path: if orders already agree, returns None too."""
    if expected_names is None or len(solver_names) != n or len(expected_names) != n:
        return None
    exp = np.asarray(expected_names, dtype=str)
    sol = np.asarray(solver_names, dtype=str)
    if np.array_equal(exp, sol):
        return None  # already in model order - positional slicing is correct
    eo = np.argsort(exp, kind="stable")
    so = np.argsort(sol, kind="stable")
    exp_sorted = exp[eo]
    if not np.array_equal(exp_sorted, sol[so]):
        return None
    if n > 1 and (exp_sorted[1:] == exp_sorted[:-1]).any():
        return None
    perm = np.empty(n, dtype=np.intp)
    perm[eo] = so
    return perm


def write_solution_csv(
    sol: Solution, directory: str | Path, use_rust: bool = True
) -> None:
    """Write solution to CSV files, one per variable and constraint."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if use_rust:
        try:
            import nimopt_rust

            _write_solution_csv_rust(sol, directory, nimopt_rust)
            return
        except ImportError:
            pass

    _write_solution_csv_python(sol, directory)


def _write_solution_csv_rust(sol: Solution, directory: Path, nimopt_rust) -> None:
    """Write solution CSVs using Rust."""
    for var_name, var_sol in sol.variables.items():
        path = str(directory / f"var_{var_name}.csv")
        nimopt_rust.write_var_solution_csv(
            path,
            var_sol.dims,
            # Stringify: set elements may be ints (e.g. integer time indices),
            # but the Rust writer takes Vec<Vec<String>>.
            [[str(e) for e in dim] for dim in var_sol.elements],
            var_sol.values.flatten().astype("float64"),
            var_sol.duals.flatten().astype("float64"),
        )

    for con_name, con_sol in sol.constraints.items():
        path = str(directory / f"con_{con_name}.csv")
        nimopt_rust.write_con_solution_csv(
            path,
            con_sol.dims,
            [[str(e) for e in dim] for dim in con_sol.elements],
            con_sol.duals.flatten().astype("float64"),
        )


def _write_solution_csv_python(sol: Solution, directory: Path) -> None:
    """Write solution CSVs using Python (fallback)."""
    # Write variables
    for var_name, var_sol in sol.variables.items():
        path = directory / f"var_{var_name}.csv"
        with open(path, "w") as f:
            header = var_sol.dims + ["value", "dual"]
            f.write(",".join(header) + "\n")

            if not var_sol.dims:
                f.write(f"{var_sol.values.flat[0]},{var_sol.duals.flat[0]}\n")
            else:
                ranges = [range(len(e)) for e in var_sol.elements]
                for idx in itertools.product(*ranges):
                    row = [var_sol.elements[d][idx[d]] for d in range(len(idx))]
                    row.append(str(var_sol.values[idx]))
                    row.append(str(var_sol.duals[idx]))
                    f.write(",".join(str(x) for x in row) + "\n")

    # Write constraints
    for con_name, con_sol in sol.constraints.items():
        path = directory / f"con_{con_name}.csv"
        with open(path, "w") as f:
            header = con_sol.dims + ["dual"]
            f.write(",".join(header) + "\n")

            if not con_sol.dims:
                f.write(f"{con_sol.duals.flat[0]}\n")
            else:
                ranges = [range(len(e)) for e in con_sol.elements]
                for idx in itertools.product(*ranges):
                    row = [con_sol.elements[d][idx[d]] for d in range(len(idx))]
                    row.append(str(con_sol.duals[idx]))
                    f.write(",".join(str(x) for x in row) + "\n")
