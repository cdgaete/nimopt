"""Direct HiGHS solver interface - no LP file generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import nimblend as nb
import numpy as np

from .base import Solver, SolverResult, SolverStatus

if TYPE_CHECKING:
    from ..model import Model
    from ..solution import Solution

try:
    import highspy

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

try:
    import nimopt_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


class HiGHSDirectSolver(Solver):
    """Direct HiGHS solver - builds model in memory without LP file."""

    def __init__(self, use_rust: bool = True):
        if not HAS_HIGHS:
            raise ImportError("highspy not installed. Run: pip install highspy")
        if use_rust and not HAS_RUST:
            from ..model import _RUST_MISSING_MSG

            raise ImportError(_RUST_MISSING_MSG)
        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", False)
        self._model: Optional[Model] = None
        self._var_names: Optional[List[str]] = None
        self._con_names: Optional[List[str]] = None
        self._var_info: Optional[List] = None  # For lazy name generation
        # use_rust=False is an explicit opt-in to the pure-Python reference
        # builder (used in tests) - it is never selected silently.
        self._use_rust = use_rust

    def read_lp(self, path: str | Path) -> None:
        self._h.readModel(str(path))

    def read_mps(self, path: str | Path) -> None:
        self._h.readModel(str(path))

    def set_option(self, name: str, value) -> None:
        self._h.setOptionValue(name, value)

    def load_model(self, model: "Model") -> None:
        self._model = model
        if self._use_rust:
            matrices = _build_matrices_rust_fast(model)
        else:
            matrices = _build_matrices(model)

        self._var_names = matrices.get("var_names")
        self._con_names = matrices.get("con_names")
        self._var_info = matrices.get("var_info")

        n_vars = matrices["n_vars"]
        self._h.addVars(n_vars, matrices["lb"], matrices["ub"])
        self._h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int32), matrices["c"])

        # Integer / binary variables (LP writer handles this via the
        # General/Binary sections; the direct path must set integrality
        # explicitly or MIPs would silently be solved as relaxations).
        int_idx = []
        offset = 0
        for var in model.variables.values():
            if var.vtype in ("integer", "binary"):
                int_idx.extend(range(offset, offset + var.size))
                if var.vtype == "binary":
                    idx = np.arange(offset, offset + var.size, dtype=np.int32)
                    self._h.changeColsBounds(
                        var.size,
                        idx,
                        np.maximum(matrices["lb"][idx], 0.0),
                        np.minimum(matrices["ub"][idx], 1.0),
                    )
            offset += var.size
        if int_idx:
            idx = np.array(int_idx, dtype=np.int32)
            kind = np.full(
                len(int_idx), highspy.HighsVarType.kInteger, dtype=np.int32
            )
            self._h.changeColsIntegrality(len(int_idx), idx, kind)

        if model.sense == "maximize":
            self._h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        if matrices["n_cons"] > 0:
            self._h.addRows(
                matrices["n_cons"],
                matrices["row_lower"],
                matrices["row_upper"],
                matrices["nnz"],
                matrices["indptr"],
                matrices["indices"],
                matrices["data"],
            )

    def solve(self) -> SolverResult:
        self._h.run()
        status = self._map_status(self._h.getModelStatus())
        info = self._h.getInfo()
        return SolverResult(
            status=status,
            objective_value=info.objective_function_value,
            solve_time=self._h.getRunTime(),
            iterations=info.simplex_iteration_count,
            nodes=info.mip_node_count,
            gap=info.mip_gap if info.mip_node_count > 0 else None,
        )

    def _map_status(self, highs_status) -> SolverStatus:
        s = str(highs_status)
        if "Optimal" in s:
            return SolverStatus.OPTIMAL
        elif "Infeasible" in s:
            return SolverStatus.INFEASIBLE
        elif "Unbounded" in s:
            return SolverStatus.UNBOUNDED
        elif "Time" in s:
            return SolverStatus.TIME_LIMIT
        elif "Iteration" in s:
            return SolverStatus.ITERATION_LIMIT
        return SolverStatus.UNKNOWN

    def get_variable_names(self) -> List[str]:
        """Get variable names (generated lazily if needed)."""
        if self._var_names is None and self._var_info is not None:
            self._var_names = _generate_var_names_from_info(self._var_info)
        return self._var_names or []

    def get_constraint_names(self) -> List[str]:
        return self._con_names or []

    def get_variable_values(self) -> List[float]:
        return self._h.allVariableValues()

    def get_variable_duals(self) -> List[float]:
        return self._h.allVariableDuals()

    def get_constraint_duals(self) -> List[float]:
        return self._h.allConstrDuals()

    def get_solution(self) -> "Solution":
        """Extract solution as nimblend Arrays.

        Returns a Solution object with .var(name) and .con(name) methods
        that return nimblend Arrays.
        """
        from ..solution import extract_solution_python

        if self._model is None:
            raise RuntimeError("No model loaded")
        return extract_solution_python(self, self._model)

    def write_solution(self, path: str | Path) -> None:
        self._h.writeSolution(str(path), 0)


def _generate_var_names_from_info(var_info: List) -> List[str]:
    """Generate variable names from var_info (lazy generation)."""
    names = []
    for var_name, dim_elements in var_info:
        if not dim_elements:
            names.append(var_name)
        else:
            names.extend(nimopt_rust.generate_var_names(var_name, dim_elements))
    return names


def _build_matrices_rust_fast(model: "Model") -> Dict:
    """Build matrices with minimal Python overhead - Rust fast path."""
    from ..expression import LinearExpr
    from ..variable import Variable, VarRef

    # === Build variable info (no names, no dict) ===
    var_start_idx: Dict[str, int] = {}
    var_info: List[Tuple[str, List[List[str]]]] = []  # For lazy name gen
    n_vars = 0

    for var in model.variables.values():
        var_start_idx[var.name] = n_vars
        if not var.sets:
            var_info.append((var.name, []))
            n_vars += 1
        else:
            dim_elements = [[str(e) for e in s.elements] for s in var.sets]
            var_info.append((var.name, dim_elements))
            n_vars += var.size

    # === Build objective vector (vectorized) ===
    c = np.zeros(n_vars, dtype=np.float64)
    if model.objective:
        from ..writers.lp_rust import _coef_flat_for_var

        for var, coef, fixed, _lagged in model.objective.terms:
            start = var_start_idx[var.name]
            n = var.size
            c[start : start + n] += _coef_flat_for_var(coef, var)

    # === Build variable bounds (vectorized) ===
    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    for var in model.variables.values():
        start = var_start_idx[var.name]
        n = var.size
        if var.lb is not None:
            lb[start : start + n] = var.lb
        if var.ub is not None:
            ub[start : start + n] = var.ub

    # === Build constraints (Rust fast path) ===
    all_indptr = [0]
    all_indices = []
    all_data = []
    all_row_lower = []
    all_row_upper = []
    con_names: List[str] = []
    total_nnz = 0

    # Name->column map for the non-vectorized fallback path. Built lazily on the
    # first fallback constraint and scoped to THIS build, so it always reflects
    # the model's current variable layout (a persistent, mutated model must not
    # reuse a column map from a previous build with different variables).
    var_idx: Optional[Dict[str, int]] = None

    for eq_name, con in model._constraints.items():
        lhs_terms = list(con.lhs.terms)
        rhs_const = 0.0
        rhs_const_array = None  # Track Array constant separately

        if isinstance(con.rhs, LinearExpr):
            for var, coef, fixed, lagged in con.rhs.terms:
                lhs_terms.append((var, _negate_coef(coef), fixed, lagged))
            if isinstance(con.rhs.const, (int, float)):
                rhs_const = con.rhs.const  # Don't negate - stays on RHS
            elif hasattr(con.rhs.const, "array"):
                # Array constant (e.g., Load[Hours]) - keep for later
                rhs_const_array = con.rhs.const.array
            elif hasattr(con.rhs.const, "values"):
                # nimblend Array - keep for later
                rhs_const_array = con.rhs.const
        elif isinstance(con.rhs, VarRef):
            # VarRef may have fixed/lagged indices
            lhs_terms.append((
                con.rhs.var,
                -1.0,
                con.rhs.fixed_indices,
                con.rhs.lagged_indices,
            ))
        elif isinstance(con.rhs, Variable):
            lhs_terms.append((con.rhs, -1.0, [], []))
        elif isinstance(con.rhs, (int, float)):
            rhs_const = float(con.rhs)
        elif hasattr(con.rhs, "array"):
            # Bare param on the RHS, e.g. `... == c[I]`.
            rhs_const_array = con.rhs.array
        elif hasattr(con.rhs, "values"):
            rhs_const_array = con.rhs

        # A constant on the LHS (e.g. from `param[I] + x[I] == ...`, where the
        # param folds into the LHS expression's constant) moves to the RHS.
        # `lhs_has_const` disqualifies the single-term fast path, which cannot
        # combine a LHS constant with the RHS.
        lhs_const = con.lhs.const
        lhs_has_const = not (isinstance(lhs_const, (int, float)) and lhs_const == 0)
        if isinstance(lhs_const, (int, float)):
            rhs_const -= lhs_const
        else:
            lhs_arr = getattr(lhs_const, "array", lhs_const)
            if hasattr(lhs_arr, "values"):
                if rhs_const_array is None:
                    rhs_const_array = lhs_arr * -1.0
                else:
                    rhs_const_array = rhs_const_array - lhs_arr

        free_sets = con.free_sets
        sense = con.sense

        # Check if any term has lagged indices
        has_lagged = any(term[3] for term in lhs_terms)

        # A duplicate column index within one CSR row can only arise when two
        # terms reference the same variable (same var -> same column block),
        # e.g. Sum(BND, b[BND]) - b[opc], or x[I] + x[I]. This cheap O(terms)
        # gate lets clean constraints skip row canonicalization entirely; only
        # the same-variable-twice family pays the (vectorized) merge cost.
        may_dupe = len({t[0].name for t in lhs_terms}) != len(lhs_terms)

        # Rust fast path: single indexed var, has free sets, no fixed/lagged indices.
        # Requires every free set to be one of the variable's own dims -- this
        # builder derives the constraint layout from the variable's shape, so a
        # free axis that lives only in the coefficient (e.g. Sum(j, a[i,j]*x[j]),
        # free i) is not representable here and is routed to the multi-term path.
        can_use_rust = (
            HAS_RUST
            and len(lhs_terms) == 1
            and lhs_terms[0][0].sets
            and free_sets
            and not lhs_has_const  # fast path can't fold a LHS constant into RHS
            and not lhs_terms[0][2]  # no fixed indices
            and not lhs_terms[0][3]  # no lagged indices
            and all(
                id(fs) in {id(s) for s in lhs_terms[0][0].sets} for fs in free_sets
            )
        )

        if can_use_rust:
            var, coef, fixed, _lagged = lhs_terms[0]
            result = _build_sum_csr_rust_fast(
                var, coef, free_sets, con.rhs, rhs_const, sense, var_start_idx[var.name]
            )
            if result is not None:
                indptr, indices, data, row_lower, row_upper, suffixes = result
                for ptr in indptr[1:]:
                    all_indptr.append(ptr + total_nnz)
                all_indices.append(indices)
                all_data.append(data)
                all_row_lower.append(row_lower)
                all_row_upper.append(row_upper)
                con_names.extend(eq_name + "_" + s for s in suffixes)
                total_nnz += len(indices)
                continue

        # Multi-term Rust fast path: multiple terms, free sets, no lagged indices
        can_use_multi_rust = (
            HAS_RUST
            and free_sets
            and not has_lagged
            and all(not term[2] for term in lhs_terms)  # no fixed indices
            and all(term[0].sets for term in lhs_terms)  # all vars indexed
        )

        if can_use_multi_rust:
            result = _build_multi_term_csr_rust(
                lhs_terms, free_sets, con.rhs, rhs_const, rhs_const_array,
                sense, var_start_idx,
            )
            if result is not None:
                indptr, indices, data, row_lower, row_upper = result
                if may_dupe:
                    indices, data, indptr = _canonicalize_csr_block(
                        indices, data, indptr
                    )
                n_rows = len(indptr) - 1
                for ptr in indptr[1:]:
                    all_indptr.append(ptr + total_nnz)
                all_indices.append(indices)
                all_data.append(data)
                all_row_lower.append(row_lower)
                all_row_upper.append(row_upper)
                # Name rows by element (C-order over free sets), matching every
                # other build path and the element-based names the solution
                # extractor expects. Positional names (bal_0, bal_1, ...) broke
                # dual extraction for multi-term (e.g. equality) constraints.
                import itertools

                from ..writers import sanitize_lp_name

                combos = list(itertools.product(*(s.elements for s in free_sets)))
                if len(combos) == n_rows:
                    con_names.extend(
                        eq_name
                        + "_"
                        + "_".join(sanitize_lp_name(e) for e in combo)
                        for combo in combos
                    )
                else:
                    con_names.extend(f"{eq_name}_{i}" for i in range(n_rows))
                total_nnz += len(indices)
                continue

        # Vectorized path for lagged constraints
        if has_lagged and free_sets:
            result = _build_lagged_constraint_vectorized(
                lhs_terms, free_sets, con.rhs, rhs_const, sense,
                var_start_idx, eq_name, rhs_const_array=rhs_const_array,
            )
            if result is not None:
                indptr, indices, data, row_lower, row_upper, names = result
                if may_dupe:
                    indices, data, indptr = _canonicalize_csr_block(
                        indices, data, indptr
                    )
                for ptr in indptr[1:]:
                    all_indptr.append(ptr + total_nnz)
                all_indices.append(indices)
                all_data.append(data)
                all_row_lower.append(row_lower)
                all_row_upper.append(row_upper)
                con_names.extend(names)
                total_nnz += len(indices)
                continue

        # Fallback needs var_idx - build it lazily, once per build
        if var_idx is None:
            var_idx = _build_var_idx(model, var_start_idx)

        if not free_sets:
            result = _expand_constraint(lhs_terms, {}, var_idx)
            if result is not None:
                idx_list, val_list = result
                rhs_val = _get_rhs_value(con.rhs, {}, rhs_const)
                total_nnz += len(idx_list)
                all_indptr.append(total_nnz)
                all_indices.append(np.array(idx_list, dtype=np.int32))
                all_data.append(np.array(val_list, dtype=np.float64))
                _append_bounds(all_row_lower, all_row_upper, sense, rhs_val)
                con_names.append(eq_name)
        else:
            import itertools

            for combo in itertools.product(*(s.elements for s in free_sets)):
                bindings = dict(zip(free_sets, combo))
                result = _expand_constraint(lhs_terms, bindings, var_idx)
                if result is None:
                    # Skip constraint (lagged index out of bounds)
                    continue
                idx_list, val_list = result
                rhs_val = _get_rhs_value(con.rhs, bindings, rhs_const)
                total_nnz += len(idx_list)
                all_indptr.append(total_nnz)
                all_indices.append(np.array(idx_list, dtype=np.int32))
                all_data.append(np.array(val_list, dtype=np.float64))
                _append_bounds(all_row_lower, all_row_upper, sense, rhs_val)
                con_names.append(eq_name + "_" + "_".join(str(e) for e in combo))

    # Combine
    n_cons = len(all_indptr) - 1
    if n_cons > 0:
        indptr = np.array(all_indptr, dtype=np.int32)
        indices = (
            np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
        )
        data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float64)
        row_lower = np.concatenate(all_row_lower)
        row_upper = np.concatenate(all_row_upper)
    else:
        indptr = np.array([0], dtype=np.int32)
        indices = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.float64)
        row_lower = np.array([], dtype=np.float64)
        row_upper = np.array([], dtype=np.float64)

    return {
        "n_vars": n_vars,
        "n_cons": n_cons,
        "var_info": var_info,
        "con_names": con_names,
        "c": c,
        "lb": lb,
        "ub": ub,
        "row_lower": row_lower,
        "row_upper": row_upper,
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "nnz": len(indices),
    }


def _build_var_idx(model, var_start_idx) -> Dict[str, int]:
    """Build a name->column map for the current variable layout.

    Not cached across builds: the caller scopes it to a single matrix build so a
    persistent, mutated model never reuses a column map from a previous build.
    """
    import itertools

    var_idx = {}
    for var in model.variables.values():
        start = var_start_idx[var.name]
        if not var.sets:
            var_idx[var.name] = start
        else:
            idx = start
            for combo in itertools.product(*(s.elements for s in var.sets)):
                vname = var.name + "_" + "_".join(str(e) for e in combo)
                var_idx[vname] = idx
                idx += 1

    return var_idx


def _build_sum_csr_rust_fast(
    var, coef, free_sets, rhs_orig, rhs_const, sense, var_start_idx
) -> Optional[Tuple]:
    """Build sum constraint CSR using fast Rust path."""
    import itertools

    import nimblend as nb

    dim_sizes = [len(s) for s in var.sets]
    free_set_ids = {id(s) for s in free_sets}
    is_free_dim = [id(s) in free_set_ids for s in var.sets]

    # Get coefficient array
    if isinstance(coef, nb.Array):
        coef_flat = coef.values.flatten().astype(np.float64)
    elif isinstance(coef, (int, float)):
        coef_flat = None
    elif hasattr(coef, "array"):
        coef_flat = coef.array.values.flatten().astype(np.float64)
    else:
        coef_flat = None

    # Get RHS array - broadcast/align to free set order
    from ..writers.lp_rust import _rhs_flat_for_free_sets

    rhs_flat = _rhs_flat_for_free_sets(rhs_orig, rhs_const, free_sets)

    sense_str = "<=" if sense == "<=" else (">=" if sense == ">=" else "=")

    indptr, indices, data, row_lower, row_upper = nimopt_rust.build_sum_csr_fast(
        var_start_idx, dim_sizes, is_free_dim, coef_flat, rhs_flat, sense_str
    )

    # Build constraint name suffixes
    suffixes = []
    for combo in itertools.product(*(s.elements for s in free_sets)):
        suffixes.append("_".join(str(e) for e in combo))

    return (
        np.asarray(indptr),
        np.asarray(indices),
        np.asarray(data),
        np.asarray(row_lower),
        np.asarray(row_upper),
        suffixes,
    )


def _build_multi_term_csr_rust(
    lhs_terms, free_sets, rhs_orig, rhs_const, rhs_const_array, sense, var_start_idx
) -> Optional[Tuple]:
    """Build multi-term constraint CSR using Rust."""
    import nimblend as nb

    free_set_ids = {id(s): i for i, s in enumerate(free_sets)}

    # Build term specs
    term_var_starts = []
    term_dim_sizes = []
    term_coefs = []
    term_coef_sizes = []      # shape of coefficient array
    term_coef_free_map = []   # maps coef dim -> free set index (-1 if not free)
    term_coef_sum_map = []    # maps coef dim -> variable summed-axis position (-1 if not summed)
    term_is_free_dims = []

    free_set_names = {fs.name: i for i, fs in enumerate(free_sets)}

    for var, coef, fixed, lagged in lhs_terms:
        term_var_starts.append(var_start_idx[var.name])
        term_dim_sizes.append([len(s) for s in var.sets])
        term_is_free_dims.append([id(s) in free_set_ids for s in var.sets])

        # Position of each of the variable's summed axes, in variable-dim order.
        # This must match the sum_combo ordering used by build_multi_term_csr,
        # which iterates the variable's dims and pushes the non-free ones.
        sum_pos_by_name = {}
        for s in var.sets:
            if id(s) not in free_set_ids:
                sum_pos_by_name[s.name] = len(sum_pos_by_name)

        def _sum_map_for_dims(dims):
            # A coef dim is a summed axis iff it names one of the variable's
            # summed sets (and is not itself a free set).
            return [
                sum_pos_by_name[d] if (d not in free_set_names and d in sum_pos_by_name) else -1
                for d in dims
            ]

        # Handle coefficient - keep it in its original shape for proper indexing
        if isinstance(coef, nb.Array):
            term_coefs.append(coef.values.flatten().astype(np.float64))
            term_coef_sizes.append(list(coef.shape))
            # Map each coef dimension to free set index
            coef_free_map = []
            for dim_name in coef.dims:
                # Find which free set this dimension corresponds to
                found = False
                for i, fs in enumerate(free_sets):
                    if fs.name == dim_name:
                        coef_free_map.append(i)
                        found = True
                        break
                if not found:
                    coef_free_map.append(-1)  # Not a free set dimension
            term_coef_free_map.append(coef_free_map)
            term_coef_sum_map.append(_sum_map_for_dims(coef.dims))
        elif hasattr(coef, "array"):
            arr = coef.array
            term_coefs.append(arr.values.flatten().astype(np.float64))
            term_coef_sizes.append(list(arr.shape))
            coef_free_map = []
            for dim_name in arr.dims:
                found = False
                for i, fs in enumerate(free_sets):
                    if fs.name == dim_name:
                        coef_free_map.append(i)
                        found = True
                        break
                if not found:
                    coef_free_map.append(-1)
            term_coef_free_map.append(coef_free_map)
            term_coef_sum_map.append(_sum_map_for_dims(arr.dims))
        elif isinstance(coef, (int, float)):
            size = 1
            for s in var.sets:
                size *= len(s)
            term_coefs.append(np.full(size, float(coef), dtype=np.float64))
            term_coef_sizes.append([len(s) for s in var.sets])
            # For scalar coef broadcast to var shape, map var dims to free sets
            coef_free_map = []
            for s in var.sets:
                if id(s) in free_set_ids:
                    coef_free_map.append(free_set_ids[id(s)])
                else:
                    coef_free_map.append(-1)
            term_coef_free_map.append(coef_free_map)
            # Scalar coef is broadcast to var shape (every entry identical), so
            # the summed axis need not be indexed -- all -1.
            term_coef_sum_map.append([-1] * len(var.sets))
        else:
            term_coefs.append(None)
            term_coef_sizes.append([])
            term_coef_free_map.append([])
            term_coef_sum_map.append([])

    # Free set sizes
    free_set_sizes = [len(s) for s in free_sets]

    # RHS array - broadcast/align to free set order
    # Priority: rhs_const_array > rhs_orig > rhs_const
    from ..writers.lp_rust import _rhs_flat_for_free_sets

    base_const = rhs_const if isinstance(rhs_const, (int, float)) else 0.0
    rhs_flat = _rhs_flat_for_free_sets(
        rhs_orig, base_const, free_sets, rhs_const_array=rhs_const_array
    )

    sense_str = "<=" if sense == "<=" else (">=" if sense == ">=" else "=")

    indptr, indices, data, row_lower, row_upper = nimopt_rust.build_multi_term_csr(
        term_var_starts,
        term_dim_sizes,
        term_coefs,
        term_coef_sizes,
        term_coef_free_map,
        term_coef_sum_map,
        term_is_free_dims,
        free_set_sizes,
        rhs_flat,
        sense_str,
    )

    return (
        np.asarray(indptr),
        np.asarray(indices),
        np.asarray(data),
        np.asarray(row_lower),
        np.asarray(row_upper),
    )


def _build_lagged_constraint_vectorized(
    lhs_terms, free_sets, rhs_orig, rhs_const, sense, var_start_indices, eq_name,
    rhs_const_array=None,
) -> Optional[Tuple]:
    """
    Build lagged constraint CSR using vectorized nimblend operations.

    Uses shift() for non-cyclic lag and roll() for cyclic lag to compute
    valid rows, then constructs CSR matrix efficiently.

    Returns (indptr, indices, data, row_lower, row_upper, con_names) or None.
    """
    import itertools

    import nimblend as nb

    if not free_sets:
        return None

    # Compute total number of potential constraint rows
    free_sizes = [len(s) for s in free_sets]
    n_rows = 1
    for sz in free_sizes:
        n_rows *= sz

    # Compute strides for decomposing flat index into per-dimension indices
    free_strides = []
    stride = 1
    for sz in reversed(free_sizes):
        free_strides.insert(0, stride)
        stride *= sz

    # Track valid rows (start with all valid, AND with each non-cyclic lag)
    valid_mask = np.ones(n_rows, dtype=bool)

    # For each term, collect info needed to compute variable indices
    # (var_start, coef_flat, var_strides, dim_to_free_pos, lag_offsets_by_dim)
    term_data = []

    for var, coef, fixed, lagged in lhs_terms:
        if not var.sets:
            # Scalar variable
            c = float(coef) if isinstance(coef, (int, float)) else 1.0
            term_data.append({
                "var_start": var_start_indices[var.name],
                "coef": c,
                "is_scalar": True,
            })
            continue

        var_start = var_start_indices[var.name]
        var_sets = var.sets

        # Build coefficient array
        if isinstance(coef, nb.Array):
            coef_vals = coef.values.flatten().astype(np.float64)
        elif isinstance(coef, (int, float)):
            coef_vals = float(coef)
        elif hasattr(coef, "array"):
            coef_vals = coef.array.values.flatten().astype(np.float64)
        else:
            coef_vals = 1.0

        # Compute variable strides
        var_strides = []
        stride = 1
        for s in reversed(var_sets):
            var_strides.insert(0, stride)
            stride *= len(s)

        # Map variable dimensions to free set positions
        dim_to_free_pos = {}
        for i, s in enumerate(var_sets):
            for j, fs in enumerate(free_sets):
                if s is fs or s.name == fs.name:
                    dim_to_free_pos[i] = j
                    break

        # Process lagged indices
        lagged_map = {pos: ls for pos, ls in lagged} if lagged else {}
        lag_offsets = {}  # dim_pos -> offset

        for pos, ls in lagged_map.items():
            offset = ls.offset  # negative for lag, positive for lead
            lag_offsets[pos] = offset

            if not ls.cyclic:
                # Mark invalid rows: where lagged index would be out of bounds
                dim_size = len(var_sets[pos])
                free_pos = dim_to_free_pos.get(pos)
                if free_pos is not None:
                    # Build mask for this dimension
                    for flat_idx in range(n_rows):
                        # Decompose flat_idx
                        fs = free_strides[free_pos]
                        dim_idx = (flat_idx // fs) % free_sizes[free_pos]
                        lagged_idx = dim_idx + offset
                        if lagged_idx < 0 or lagged_idx >= dim_size:
                            valid_mask[flat_idx] = False

        term_data.append({
            "var_start": var_start,
            "coef": coef_vals,
            "is_scalar": False,
            "var_strides": var_strides,
            "dim_to_free_pos": dim_to_free_pos,
            "lag_offsets": lag_offsets,
            "var_sizes": [len(s) for s in var_sets],
        })

    # Build CSR matrix for valid rows only
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        return None

    # Pre-allocate CSR arrays
    indptr = np.zeros(n_valid + 1, dtype=np.int32)
    indices_list = []
    data_list = []

    # Build RHS array - broadcast/align to free set order
    from ..writers.lp_rust import _rhs_flat_for_free_sets

    rhs_flat = _rhs_flat_for_free_sets(
        rhs_orig, rhs_const, free_sets, rhs_const_array=rhs_const_array
    )

    # Fill CSR arrays
    row_lower = np.zeros(n_valid, dtype=np.float64)
    row_upper = np.zeros(n_valid, dtype=np.float64)

    ptr = 0
    for out_row, flat_idx in enumerate(valid_indices):
        indptr[out_row] = ptr

        # Set bounds
        rhs_val = rhs_flat[flat_idx] if len(rhs_flat) > 1 else rhs_flat[0]
        if sense == "<=":
            row_lower[out_row] = -np.inf
            row_upper[out_row] = rhs_val
        elif sense == ">=":
            row_lower[out_row] = rhs_val
            row_upper[out_row] = np.inf
        else:
            row_lower[out_row] = rhs_val
            row_upper[out_row] = rhs_val

        # Decompose flat_idx into per-dimension indices
        dim_indices = []
        remaining = flat_idx
        for stride in free_strides:
            dim_indices.append(remaining // stride)
            remaining = remaining % stride

        for td in term_data:
            if td["is_scalar"]:
                indices_list.append(td["var_start"])
                data_list.append(td["coef"])
                ptr += 1
            else:
                # Compute variable flat index
                var_flat_idx = 0
                for var_dim, var_stride in enumerate(td["var_strides"]):
                    free_pos = td["dim_to_free_pos"].get(var_dim)
                    if free_pos is not None:
                        idx = dim_indices[free_pos]
                        # Apply lag offset
                        offset = td["lag_offsets"].get(var_dim, 0)
                        idx = idx + offset
                        # Handle cyclic wrap (if cyclic, valid_mask didn't filter)
                        idx = idx % td["var_sizes"][var_dim]
                        var_flat_idx += idx * var_stride

                # Get coefficient
                coef = td["coef"]
                if isinstance(coef, np.ndarray):
                    c = coef[flat_idx] if flat_idx < len(coef) else coef[0]
                else:
                    c = coef

                if c != 0:
                    indices_list.append(td["var_start"] + var_flat_idx)
                    data_list.append(c)
                    ptr += 1

    indptr[n_valid] = ptr

    # Convert to arrays
    indices = np.array(indices_list, dtype=np.int32)
    data = np.array(data_list, dtype=np.float64)

    # Build constraint names for valid rows
    con_names = []
    all_combos = list(itertools.product(*(s.elements for s in free_sets)))
    for flat_idx in valid_indices:
        combo = all_combos[flat_idx]
        con_names.append(eq_name + "_" + "_".join(str(e) for e in combo))

    return (indptr, indices, data, row_lower, row_upper, con_names)


def _append_bounds(row_lower, row_upper, sense, rhs):
    if sense == "<=":
        row_lower.append(np.array([-np.inf], dtype=np.float64))
        row_upper.append(np.array([rhs], dtype=np.float64))
    elif sense == ">=":
        row_lower.append(np.array([rhs], dtype=np.float64))
        row_upper.append(np.array([np.inf], dtype=np.float64))
    else:
        row_lower.append(np.array([rhs], dtype=np.float64))
        row_upper.append(np.array([rhs], dtype=np.float64))


def _build_matrices(model: "Model") -> Dict:
    """Build matrices (pure Python fallback)."""
    import itertools

    from ..expression import LinearExpr
    from ..variable import Variable, VarRef

    var_idx: Dict[str, int] = {}
    var_names: List[str] = []
    n_vars = 0

    for var in model.variables.values():
        if not var.sets:
            var_idx[var.name] = n_vars
            var_names.append(var.name)
            n_vars += 1
        else:
            for combo in itertools.product(*(s.elements for s in var.sets)):
                vname = var.name + "_" + "_".join(str(e) for e in combo)
                var_idx[vname] = n_vars
                var_names.append(vname)
                n_vars += 1

    c = np.zeros(n_vars, dtype=np.float64)
    if model.objective:
        for var, coef, fixed, _lagged in model.objective.terms:
            if not var.sets:
                c[var_idx[var.name]] += _get_scalar_coef(coef)
            else:
                for i, combo in enumerate(
                    itertools.product(*(s.elements for s in var.sets))
                ):
                    vname = var.name + "_" + "_".join(str(e) for e in combo)
                    c[var_idx[vname]] += _get_coef_for_combo(coef, var.sets, combo)

    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    for var in model.variables.values():
        for vname in var.all_names():
            idx = var_idx[vname]
            if var.lb is not None:
                lb[idx] = var.lb
            if var.ub is not None:
                ub[idx] = var.ub

    row_data = []
    con_names = []
    for eq_name, con in model._constraints.items():
        lhs_terms = list(con.lhs.terms)
        rhs_const = 0.0
        rhs_const_array = None  # Track array const separately

        if isinstance(con.rhs, LinearExpr):
            for var, coef, fixed, lagged in con.rhs.terms:
                lhs_terms.append((var, _negate_coef(coef), fixed, lagged))
            if isinstance(con.rhs.const, (int, float)):
                rhs_const = con.rhs.const  # Don't negate - stays on RHS
            elif isinstance(con.rhs.const, nb.Array):
                rhs_const_array = con.rhs.const  # Keep the array for later
        elif isinstance(con.rhs, VarRef):
            # VarRef may have fixed/lagged indices
            lhs_terms.append((
                con.rhs.var,
                -1.0,
                con.rhs.fixed_indices,
                con.rhs.lagged_indices,
            ))
        elif isinstance(con.rhs, Variable):
            lhs_terms.append((con.rhs, -1.0, [], []))
        elif isinstance(con.rhs, (int, float)):
            rhs_const = float(con.rhs)
        elif hasattr(con.rhs, "array"):
            # Bare param on the RHS, e.g. `... == c[I]`.
            rhs_const_array = con.rhs.array
        elif hasattr(con.rhs, "values"):
            rhs_const_array = con.rhs

        # A constant on the LHS (e.g. from `param[I] + x[I] == ...`, where the
        # param folds into the LHS expression's constant) moves to the RHS.
        # `lhs_has_const` disqualifies the single-term fast path, which cannot
        # combine a LHS constant with the RHS.
        lhs_const = con.lhs.const
        lhs_has_const = not (isinstance(lhs_const, (int, float)) and lhs_const == 0)
        if isinstance(lhs_const, (int, float)):
            rhs_const -= lhs_const
        else:
            lhs_arr = getattr(lhs_const, "array", lhs_const)
            if hasattr(lhs_arr, "values"):
                if rhs_const_array is None:
                    rhs_const_array = lhs_arr * -1.0
                else:
                    rhs_const_array = rhs_const_array - lhs_arr

        free_sets = con.free_sets
        sense = con.sense

        if not free_sets:
            result = _expand_constraint(lhs_terms, {}, var_idx)
            if result is not None:
                idx_list, val_list = result
                if rhs_const_array is not None:
                    rhs_val = float(rhs_const_array.values.flat[0])
                else:
                    rhs_val = _get_rhs_value(con.rhs, {}, rhs_const)
                row_data.append((idx_list, val_list, sense, rhs_val))
                con_names.append(eq_name)
        else:
            for combo in itertools.product(*(s.elements for s in free_sets)):
                bindings = dict(zip(free_sets, combo))
                result = _expand_constraint(lhs_terms, bindings, var_idx)
                if result is None:
                    continue
                idx_list, val_list = result
                if rhs_const_array is not None:
                    # Index into the array using bindings
                    rhs_val = _get_array_value_for_bindings(
                        rhs_const_array, free_sets, combo
                    )
                else:
                    rhs_val = _get_rhs_value(con.rhs, bindings, rhs_const)
                row_data.append((idx_list, val_list, sense, rhs_val))
                con_names.append(eq_name + "_" + "_".join(str(e) for e in combo))

    n_cons = len(row_data)
    nnz = sum(len(r[0]) for r in row_data)
    indptr = np.zeros(n_cons + 1, dtype=np.int32)
    indices = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float64)
    row_lower = np.zeros(n_cons, dtype=np.float64)
    row_upper = np.zeros(n_cons, dtype=np.float64)

    ptr = 0
    for i, (col_idx, col_val, sense, rhs) in enumerate(row_data):
        indptr[i] = ptr
        for j, (ci, vi) in enumerate(zip(col_idx, col_val)):
            indices[ptr + j] = ci
            data[ptr + j] = vi
        ptr += len(col_idx)
        if sense == "<=":
            row_lower[i] = -np.inf
            row_upper[i] = rhs
        elif sense == ">=":
            row_lower[i] = rhs
            row_upper[i] = np.inf
        else:
            row_lower[i] = rhs
            row_upper[i] = rhs
    indptr[n_cons] = ptr

    return {
        "n_vars": n_vars,
        "n_cons": n_cons,
        "var_names": var_names,
        "con_names": con_names,
        "c": c,
        "lb": lb,
        "ub": ub,
        "row_lower": row_lower,
        "row_upper": row_upper,
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "nnz": nnz,
    }


def _canonicalize_csr_block(indices, data, indptr):
    """Merge duplicate column indices within each row of a CSR block.

    Two terms that reference the same variable can emit the same solver column
    into one constraint row -- e.g. a Sum over a set plus a literal element of
    that set (``Sum(BND, b[BND]) - b[opc]``), or ``x[I] + x[I]``. HiGHS rejects
    a row that repeats a column index: it silently drops the whole row, which
    later surfaces as a length-0 dual slice and a ``reshape(0 -> (n,))`` crash
    in extract_solution. Summing the contributions per (row, column) and
    dropping exact zeros yields a row with unique columns; a row that fully
    cancels becomes a valid empty row (a real ``0 <= rhs`` constraint).

    This is the single canonicalization applied to every block-producing path
    (Rust single-sum / multi-term, vectorized-lagged); the per-row
    ``_expand_constraint`` fallback accumulates per column directly. Callers
    gate on a cheap "does any variable appear in >=2 terms" test, so clean
    constraints never reach here. When they do, the work is fully vectorized
    (no per-row Python loop): rows are encoded into a single (row, column) key,
    ``np.unique`` + ``np.add.at`` accumulates, and the row pointer is rebuilt
    from per-row counts. The row count is preserved so row bounds/names stay
    aligned.
    """
    indices = np.asarray(indices)
    data = np.asarray(data, dtype=np.float64)
    indptr = np.asarray(indptr)
    n_rows = len(indptr) - 1
    if n_rows <= 0 or indices.size == 0:
        return indices.astype(np.int32), data, indptr.astype(np.int32)

    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
    ncol = int(indices.max()) + 1
    keys = row_ids * np.int64(ncol) + indices.astype(np.int64)
    uk, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)  # some numpy versions return a column vector
    acc = np.zeros(len(uk), dtype=np.float64)
    np.add.at(acc, inv, data)

    nz = acc != 0.0
    uk = uk[nz]
    acc = acc[nz]
    u_row = uk // ncol
    u_col = (uk % ncol).astype(np.int32)

    counts = np.bincount(u_row, minlength=n_rows)
    new_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return u_col, acc, new_indptr


def _expand_constraint(lhs_terms, bindings, var_idx):
    """Expand constraint terms for given bindings.

    Returns (indices, values) or None if constraint should be skipped
    (e.g., lagged index out of bounds).
    """
    import itertools

    # First pass: check if any lagged indices are out of bounds
    for var, coef, fixed, lagged in lhs_terms:
        if lagged:
            for pos, ls in lagged:
                # Find the current element at this position from bindings
                base_set = ls.base_set
                if base_set in bindings:
                    elem = bindings[base_set]
                    lagged_elem = ls.get_lagged_element(elem)
                    if lagged_elem is None:
                        # Out of bounds - skip entire constraint
                        return None

    # Second pass: build constraint row.
    #
    # Accumulate coefficients into a per-column structure keyed by the
    # resolved column index. A single constraint row can reference the same
    # column from more than one term -- e.g. a Sum expanding b[BND] over the
    # whole set contributes b_opc, and a literal b[opc] term contributes it
    # again. HiGHS rejects a CSR row that repeats a column index (it silently
    # drops the whole row -> numRow mismatch -> a length-0 dual slice blows up
    # in extract_solution). Summing every contribution per column first, and
    # only dropping exact zeros at the very end, makes cancellation (+1, -1 ->
    # dropped), doubling (+1, +1 -> +2) and single-column-meets-block all fall
    # out of one accumulation with no duplicate index ever reaching the solver.
    # dict preserves first-seen column order, which CSR does not require to be
    # sorted -- only unique.
    col_coeffs: Dict[int, float] = {}

    def _accumulate(col_idx: int, cv: float) -> None:
        col_coeffs[col_idx] = col_coeffs.get(col_idx, 0.0) + cv

    for var, coef, fixed, lagged in lhs_terms:
        if not var.sets:
            # Binding-aware: a scalar variable may still carry an indexed
            # coefficient (e.g. capb[BND]*TB inside a free-BND constraint).
            # Resolve the coef against the current free-set bindings instead
            # of collapsing to its first element.
            _accumulate(var_idx[var.name], _get_coef_for_bindings(coef, bindings))
        else:
            # Build lagged_map: position -> LaggedSet for quick lookup
            lagged_map = {pos: ls for pos, ls in lagged} if lagged else {}
            # Honor fixed (literal) indices, e.g. b[silica]: pin that
            # dimension to the named element instead of expanding over the set.
            fixed_map = {pos: val for pos, val in fixed} if fixed else {}

            var_combos = []
            for i, s in enumerate(var.sets):
                if i in fixed_map:
                    elem = fixed_map[i]
                    if elem not in s.elements:
                        raise ValueError(
                            f"Fixed index '{elem}' on variable '{var.name}' "
                            f"is not an element of set '{s.name}'"
                        )
                    var_combos.append([elem])
                elif s in bindings:
                    var_combos.append([bindings[s]])
                elif i in lagged_map:
                    # For lagged dims, use the binding for base set
                    base_set = lagged_map[i].base_set
                    if base_set in bindings:
                        var_combos.append([bindings[base_set]])
                    else:
                        var_combos.append(lagged_map[i].base_set.elements)
                else:
                    var_combos.append(s.elements)

            for combo in itertools.product(*var_combos) if var_combos else [()]:
                # Apply lag offset to get actual variable indices
                actual_combo = list(combo)
                for pos, ls in lagged_map.items():
                    elem = combo[pos]
                    lagged_elem = ls.get_lagged_element(elem)
                    # Should not be None here since we checked in first pass
                    actual_combo[pos] = lagged_elem

                vname = var.name + "_" + "_".join(str(e) for e in actual_combo)
                idx = var_idx.get(vname)
                if idx is not None:
                    # Build full index dict: var_sets combo + outer bindings
                    full_bindings = dict(bindings)
                    for s, elem in zip(var.sets, combo):
                        full_bindings[s] = elem
                    cv = _get_coef_for_bindings(coef, full_bindings)
                    # Do not drop zeros here: a later term may add a nonzero
                    # contribution to this same column, or vice versa. Zeros
                    # are dropped once, after all terms are accumulated.
                    _accumulate(idx, cv)

    # Drop exact zeros only at the end so genuine cancellation collapses the
    # column instead of emitting a stale index. A row that fully cancels to
    # zero terms returns empty lists; the caller emits a valid trivial row
    # (0 <= rhs), which is a real constraint row and never a length-0 array
    # reaching a reshape.
    indices = [idx for idx, cv in col_coeffs.items() if cv != 0.0]
    values = [cv for cv in col_coeffs.values() if cv != 0.0]
    return indices, values


def _get_coef_for_bindings(coef, bindings) -> float:
    """Resolve a coefficient to a scalar for the given bindings (Set -> element).

    Delegates label selection to nimblend's Array.sel so there is a single
    selection code path shared with the rest of the stack. sel() returns a
    Python scalar when every dimension is selected and raises on a missing
    label (rather than silently returning 0.0).
    """
    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        # Map only the dims this array actually has to their bound element.
        sel = {}
        for dim in coef.dims:
            for s, elem in bindings.items():
                if s.name == dim:
                    sel[dim] = elem
                    break
        if not sel:
            # Genuinely scalar coefficient (0-d / single value).
            if coef.values.size != 1:
                raise ValueError(
                    f"Coefficient has dims {list(coef.dims)} but no matching "
                    f"free-set bindings were provided; cannot resolve to a "
                    f"scalar. This is an internal nimopt bug."
                )
            return float(coef.values.item())
        return float(coef.sel(sel))
    elif hasattr(coef, "array"):
        return _get_coef_for_bindings(coef.array, bindings)
    return float(coef)


def _get_scalar_coef(coef) -> float:
    """Resolve a genuinely scalar coefficient.

    Fail-fast: if handed a multi-element array, the caller reached here with a
    coefficient that carries unresolved indices, which previously produced a
    silently wrong constraint matrix. Raise instead so the bug is loud.
    """
    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        if coef.values.size != 1:
            raise ValueError(
                f"_get_scalar_coef received a {coef.values.shape} array with "
                f"dims {list(coef.dims)}; this term needs binding-aware "
                f"resolution via _get_coef_for_bindings. This is an internal "
                f"nimopt bug (unresolved indexed coefficient on a scalar term)."
            )
        return float(coef.values.item())
    elif hasattr(coef, "array"):
        return _get_scalar_coef(coef.array)
    return float(coef)


def _get_array_coef(coef, flat_idx) -> float:
    import nimblend as nb

    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        return float(coef.values.flat[flat_idx])
    elif hasattr(coef, "array"):
        return float(coef.array.values.flat[flat_idx])
    return float(coef)


def _get_coef_for_combo(coef, var_sets, combo) -> float:
    import nimblend as nb

    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        indices = []
        for i, s in enumerate(var_sets):
            if s.name in coef.dims:
                coord_list = list(coef.coords[s.name])
                try:
                    indices.append(coord_list.index(combo[i]))
                except ValueError:
                    return 0.0
        if indices:
            return float(coef.values[tuple(indices)])
        return float(coef.values.flat[0])
    elif hasattr(coef, "array"):
        return _get_coef_for_combo(coef.array, var_sets, combo)
    return float(coef)


def _get_rhs_value(rhs, bindings, const) -> float:
    if isinstance(rhs, (int, float)):
        # `const` already folds the numeric rhs together with any LHS constant
        # moved across (const == float(rhs) - lhs_const). Returning float(rhs)
        # here would silently drop that LHS-constant adjustment, so `z + 3 <= 9`
        # would be built as `z <= 9` instead of `z <= 6`.
        return float(const)
    elif hasattr(rhs, "array"):
        arr = rhs.array
        if not bindings:
            return float(arr.values.flat[0])
        indices = []
        for s in rhs.sets:
            if s in bindings:
                coord_list = list(arr.coords[s.name])
                indices.append(coord_list.index(bindings[s]))
        if indices:
            return float(arr.values[tuple(indices)])
        return float(arr.values.flat[0])
    elif hasattr(rhs, "values"):
        return float(rhs.values.flat[0])
    return const


def _get_array_value_for_bindings(arr, free_sets, combo) -> float:
    """Get a value from a nimblend Array using free_sets and combo bindings."""
    indices = []
    for i, s in enumerate(free_sets):
        if s.name in arr.dims:
            coord_list = list(arr.coords[s.name])
            try:
                indices.append(coord_list.index(combo[i]))
            except ValueError:
                return 0.0
    if indices:
        return float(arr.values[tuple(indices)])
    return float(arr.values.flat[0])


def _negate_coef(coef):
    import nimblend as nb

    if isinstance(coef, (int, float)):
        return -coef
    elif isinstance(coef, nb.Array):
        return coef * (-1)
    return coef
