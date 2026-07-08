"""LP writer using Rust acceleration with optimized paths."""

import itertools
from typing import List

import nimblend as nb
import numpy as np

from . import sanitize_lp_name

try:
    import nimopt_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from ..sets import Set


def write_lp_rust(model, filename: str) -> None:
    """Write LP using Rust-accelerated functions.

    Raises ImportError if nimopt_rust is not installed - no silent
    Python fallback. Use writers.lp.write_lp explicitly if needed.
    """
    if not HAS_RUST:
        from ..model import _RUST_MISSING_MSG

        raise ImportError(_RUST_MISSING_MSG)

    nimopt_rust.write_lp_header(filename, model.name, model.sense)
    _write_objective_fast(model, filename)
    nimopt_rust.write_subject_to(filename)
    _write_constraints_rust(model, filename)
    _write_bounds_fast(model, filename)
    ints, bins = _get_var_types(model)
    nimopt_rust.write_var_types(filename, ints, bins)


def _write_objective_fast(model, filename: str) -> None:
    """Write objective using Rust with name generation."""
    if not model.objective:
        # Write empty objective
        with open(filename, "a") as f:
            f.write(" obj: 0\n\n")
        return

    # Accumulate coefficients for each variable (handles duplicates)
    coef_map = {}

    for term in model.objective.terms:
        var, coef, _ = term[0], term[1], term[2]
        if var.sets:
            dim_elements = [[sanitize_lp_name(e) for e in s.elements] for s in var.sets]
            coefs_arr = _coef_flat_for_var(coef, var)

            # Generate variable names and accumulate coefficients
            var_names = nimopt_rust.generate_var_names(var.name, dim_elements)
            for vname, c in zip(var_names, coefs_arr):
                coef_map[vname] = coef_map.get(vname, 0.0) + c
        else:
            # Scalar variable
            if isinstance(coef, (int, float)):
                c = float(coef)
            elif hasattr(coef, "values"):
                c = float(coef.values.flat[0])
            else:
                c = float(coef)
            coef_map[var.name] = coef_map.get(var.name, 0.0) + c

    # Write objective with accumulated coefficients
    all_var_names = list(coef_map.keys())
    all_coefs = np.array(list(coef_map.values()), dtype=np.float64)
    nimopt_rust.write_objective(filename, all_var_names, all_coefs)


def _get_objective_data(model):
    """Extract objective variable names and coefficients."""
    var_names = []
    coefs = []
    if model.objective:
        for term in model.objective.terms:
            var, coef, _ = term[0], term[1], term[2]
            # Generate all variable names
            if var.sets:
                combos = list(itertools.product(*(s.elements for s in var.sets)))
                for combo in combos:
                    suffix = "_" + "_".join(sanitize_lp_name(e) for e in combo)
                    var_names.append(var.name + suffix)
            else:
                var_names.append(var.name)
                combos = [()]

            # Get coefficients - vectorized when possible
            if isinstance(coef, (int, float)):
                coefs.extend([float(coef)] * len(combos))
            elif isinstance(coef, nb.Array):
                # Flatten in same order as itertools.product
                coefs.extend(coef.values.flatten().tolist())
            elif hasattr(coef, "array"):
                coefs.extend(coef.array.values.flatten().tolist())
            else:
                coefs.extend([float(coef)] * len(combos))

    return var_names, np.array(coefs, dtype=np.float64)


def _get_coef_scalar(coef, var_sets: List[Set], combo: tuple) -> float:
    """Get scalar coefficient value for a variable index."""
    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        indices = []
        for i, s in enumerate(var_sets):
            if s.name in coef.dims:
                coord_list = list(coef.coords[s.name])
                indices.append(coord_list.index(combo[i]))
        if indices:
            return float(coef.values[tuple(indices)])
        return float(coef.values.flat[0])
    elif hasattr(coef, "array"):
        return _get_coef_scalar(coef.array, var_sets, combo)
    return float(coef)


def _write_constraints_rust(model, filename: str):
    """Write constraints using Rust."""
    from ..expression import LinearExpr
    from ..variable import Variable, VarRef

    for eq_name, con in model._constraints.items():
        lhs_terms = list(con.lhs.terms)
        rhs_const = 0.0

        if isinstance(con.rhs, LinearExpr):
            for term in con.rhs.terms:
                var, coef, fixed = term[0], term[1], term[2]
                lagged = term[3] if len(term) > 3 else []
                lhs_terms.append((var, _negate_coef(coef), fixed, lagged))
            if isinstance(con.rhs.const, (int, float)):
                rhs_const = -con.rhs.const
            elif isinstance(con.rhs.const, nb.Array):
                # Array const becomes the RHS (NOT negated - unlike scalar)
                rhs_const = con.rhs.const
        elif isinstance(con.rhs, (Variable, VarRef)):
            var = con.rhs if isinstance(con.rhs, Variable) else con.rhs.var
            lhs_terms.append((var, -1.0, [], []))
        elif isinstance(con.rhs, (int, float)):
            rhs_const = float(con.rhs)

        free_sets = con.free_sets
        sense = "<=" if con.sense == "<=" else (">=" if con.sense == ">=" else "=")

        if not free_sets:
            _write_single_constraint_fast(
                filename, eq_name, sense, lhs_terms, rhs_const, con.rhs
            )
        elif _can_use_sum_constraints(lhs_terms, free_sets):
            _write_sum_constraints_fast(
                filename, eq_name, sense, lhs_terms, free_sets, rhs_const, con.rhs
            )
        else:
            _write_batch_constraints_rust(
                filename,
                eq_name,
                sense,
                lhs_terms,
                free_sets,
                rhs_const,
                con.rhs,
                model,
            )


def _can_use_sum_constraints(lhs_terms, free_sets) -> bool:
    """Check if we can use the optimized sum constraints path."""
    if len(lhs_terms) != 1:
        return False
    term = lhs_terms[0]
    _, _, fixed = term[0], term[1], term[2]
    lagged = term[3] if len(term) > 3 else []
    if fixed or lagged:
        return False
    return True


def _write_sum_constraints_fast(
    filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig
):
    """Use optimized Rust path for Sum constraints."""
    term = lhs_terms[0]
    var, coef, _ = term[0], term[1], term[2]

    # Build var_dim_elements and is_free_dim in variable dimension order
    free_set_ids = {id(s) for s in free_sets}
    var_dim_elements = [[sanitize_lp_name(e) for e in s.elements] for s in var.sets]
    is_free_dim = [id(s) in free_set_ids for s in var.sets]

    # Coefficient array
    if isinstance(coef, nb.Array):
        coef_shape = list(coef.shape)
        coef_flat = coef.values.flatten().astype(np.float64)
    elif isinstance(coef, (int, float)):
        coef_flat = None
        coef_shape = []
    elif hasattr(coef, "array"):
        arr = coef.array
        coef_shape = list(arr.shape)
        coef_flat = arr.values.flatten().astype(np.float64)
    else:
        coef_flat = None
        coef_shape = []

    # RHS array - broadcast/align to free set order
    rhs_flat = _rhs_flat_for_free_sets(rhs_orig, rhs_const, free_sets)

    nimopt_rust.write_sum_constraints(
        filename,
        eq_name,
        sense,
        var.name,
        var_dim_elements,
        is_free_dim,
        coef_flat,
        coef_shape,
        rhs_flat,
    )


def _write_batch_constraints_rust(
    filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig, model
):
    """Write constraints using Rust - fast path with name generation in Rust."""
    # Try the fast path first: all terms have no lagged indices
    has_lagged = any(len(term) > 3 and term[3] for term in lhs_terms)

    if not has_lagged:
        try:
            return _write_multi_term_fast(
                filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig
            )
        except Exception:
            pass  # Fall back to slow path

    # Slow path: build variable names in Python
    return _write_batch_constraints_rust_slow(
        filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig, model
    )


def _write_multi_term_fast(
    filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig
):
    """Fast path: generate variable names in Rust."""
    free_set_ids = {id(s): i for i, s in enumerate(free_sets)}
    free_set_names = {s.name: i for i, s in enumerate(free_sets)}

    # Build term specs for Rust
    term_var_names = []
    term_dim_elements = []
    term_coefs = []
    term_is_free_dims = []
    term_coef_shapes = []
    term_coef_free_maps = []
    term_scalar_coefs = []  # Scalar coefficient for each term

    for term in lhs_terms:
        var, coef = term[0], term[1]
        term_var_names.append(var.name)

        # Dimension elements (sanitized)
        dim_elems = [[sanitize_lp_name(e) for e in s.elements] for s in var.sets]
        term_dim_elements.append(dim_elems)

        # Is each dimension free?
        is_free = [id(s) in free_set_ids for s in var.sets]
        term_is_free_dims.append(is_free)

        # Coefficient array - handle dimension mismatch
        if isinstance(coef, nb.Array):
            term_coefs.append(coef.values.flatten().astype(np.float64))
            term_coef_shapes.append(list(coef.shape))
            # Map coef dimensions to free set indices
            coef_free_map = []
            for dim in coef.dims:
                if dim in free_set_names:
                    coef_free_map.append(free_set_names[dim])
                else:
                    coef_free_map.append(-1)  # Not a free set
            term_coef_free_maps.append(coef_free_map)
            term_scalar_coefs.append(1.0)  # Not used for array coefs
        elif hasattr(coef, "array"):
            arr = coef.array
            term_coefs.append(arr.values.flatten().astype(np.float64))
            term_coef_shapes.append(list(arr.shape))
            coef_free_map = []
            for dim in arr.dims:
                if dim in free_set_names:
                    coef_free_map.append(free_set_names[dim])
                else:
                    coef_free_map.append(-1)
            term_coef_free_maps.append(coef_free_map)
            term_scalar_coefs.append(1.0)  # Not used for array coefs
        elif isinstance(coef, (int, float)):
            term_coefs.append(None)  # Scalar - use term_scalar_coefs
            term_coef_shapes.append([])
            term_coef_free_maps.append([])
            term_scalar_coefs.append(float(coef))
        else:
            term_coefs.append(None)
            term_coef_shapes.append([])
            term_coef_free_maps.append([])
            term_scalar_coefs.append(float(coef) if coef is not None else 1.0)

    # Free set sizes
    free_set_sizes = [len(s) for s in free_sets]

    # RHS array - broadcast/align to free set order
    if isinstance(rhs_const, nb.Array):
        # rhs_const is already an Array (e.g., negated demand)
        rhs_flat = _rhs_flat_for_free_sets(
            None, 0.0, free_sets, rhs_const_array=rhs_const
        )
    else:
        rhs_flat = _rhs_flat_for_free_sets(rhs_orig, float(rhs_const), free_sets)

    nimopt_rust.write_multi_term_constraints(
        filename,
        eq_name,
        sense,
        term_var_names,
        term_dim_elements,
        term_coefs,
        term_coef_shapes,
        term_coef_free_maps,
        term_is_free_dims,
        free_set_sizes,
        rhs_flat,
        term_scalar_coefs,
    )


def _write_batch_constraints_rust_slow(
    filename, eq_name, sense, lhs_terms, free_sets, rhs_const, rhs_orig, model
):
    """Slow path: build variable names in Python (for lagged indices)."""
    all_var_names = []
    all_coefs = []
    all_rhs = []
    vars_per_con = 0

    free_combos = list(itertools.product(*(s.elements for s in free_sets)))

    for free_idx, free_combo in enumerate(free_combos):
        bindings = dict(zip(free_sets, free_combo))
        con_var_names = []
        con_coefs = []

        for term in lhs_terms:
            var, coef, _ = term[0], term[1], term[2]
            # lagged = term[3] if len(term) > 3 else []  # TODO: handle lag
            var_combos = []
            for s in var.sets:
                if s in bindings:
                    var_combos.append([bindings[s]])
                else:
                    var_combos.append(s.elements)

            for combo in itertools.product(*var_combos) if var_combos else [()]:
                suffix = (
                    "_" + "_".join(sanitize_lp_name(e) for e in combo) if combo else ""
                )
                vname = var.name + suffix
                cv = _get_coef_scalar(coef, var.sets, combo)
                con_var_names.append(vname)
                con_coefs.append(cv)

        if free_idx == 0:
            vars_per_con = len(con_var_names)

        all_var_names.extend(con_var_names)
        all_coefs.extend(con_coefs)

        if isinstance(rhs_orig, nb.Array):
            # Direct nb.Array (e.g., from Param * Param)
            indices = []
            for dim in rhs_orig.dims:
                for s, elem in bindings.items():
                    if s.name == dim:
                        coord_list = list(rhs_orig.coords[dim])
                        indices.append(coord_list.index(elem))
                        break
            if indices:
                rv = float(rhs_orig.values[tuple(indices)])
            else:
                rv = float(rhs_orig.values.flat[0])
        elif hasattr(rhs_orig, "array"):
            indices = [s.index(bindings[s]) for s in rhs_orig.sets if s in bindings]
            if indices:
                rv = float(rhs_orig.array.values[tuple(indices)])
            else:
                rv = float(rhs_orig.values.flat[0])
        else:
            rv = rhs_const
        all_rhs.append(rv)

    nimopt_rust.write_coef_constraints(
        filename,
        eq_name,
        sense,
        all_var_names,
        np.array(all_coefs, dtype=np.float64),
        np.array(all_rhs, dtype=np.float64),
        vars_per_con,
    )


def _rhs_flat_for_free_sets(rhs_orig, rhs_const, free_sets, rhs_const_array=None):
    """RHS values aligned to the constraint's C-order flat row index.

    Broadcasts partial-dimension RHS arrays (e.g. cap[G] for a
    constraint free over [G, T]) to the free sets' full shape and dim
    order via nimblend before flattening. Flattening without this check
    silently mis-sizes/mis-orders row bounds and can make models
    infeasible or wrong.
    """
    n = 1
    for s in free_sets:
        n *= len(s)
    arr = None
    if rhs_const_array is not None:
        arr = (
            rhs_const_array.array
            if hasattr(rhs_const_array, "array")
            else rhs_const_array
        )
    elif hasattr(rhs_orig, "array"):
        arr = rhs_orig.array
    elif hasattr(rhs_orig, "values"):
        arr = rhs_orig
    if arr is None or not hasattr(arr, "dims"):
        return np.full(n, float(rhs_const), dtype=np.float64)
    dims = [s.name for s in free_sets]
    shape = tuple(len(s) for s in free_sets)
    if list(arr.dims) != dims or arr.shape != shape:
        coords = {s.name: np.array(s.elements) for s in free_sets}
        ones = nb.Array(np.ones(shape), coords, dims)
        arr = arr * ones
        if list(arr.dims) != dims:
            arr = arr.transpose(*dims)
    return arr.values.reshape(-1).astype(np.float64)


def _coef_flat_for_var(coef, var) -> np.ndarray:
    """Coefficient values aligned to the variable's C-order flat index.

    Broadcasts lower-dimensional coefficient arrays (e.g. ef[G] against
    p[G,T]) to the variable's full shape and dim order via nimblend
    before flattening. Assuming coef.shape == var.shape without checking
    silently corrupts constraints when a param covers only a subset of
    the variable's dimensions.
    """
    n = var.size
    if not var.sets:
        if isinstance(coef, (int, float)):
            return np.array([float(coef)])
        arr = coef.array if hasattr(coef, "array") else coef
        if hasattr(arr, "values"):
            return np.array([float(arr.values.flat[0])])
        return np.array([float(coef)])
    if isinstance(coef, (int, float)):
        return np.full(n, float(coef), dtype=np.float64)
    arr = coef.array if hasattr(coef, "array") else coef
    if not isinstance(arr, nb.Array):
        return np.full(n, float(coef), dtype=np.float64)
    var_dims = [s.name for s in var.sets]
    var_shape = tuple(len(s) for s in var.sets)
    if list(arr.dims) != var_dims or arr.shape != var_shape:
        coords = {s.name: np.array(s.elements) for s in var.sets}
        ones = nb.Array(np.ones(var_shape), coords, var_dims)
        arr = arr * ones
        if list(arr.dims) != var_dims:
            arr = arr.transpose(*var_dims)
    return arr.values.reshape(-1).astype(np.float64)


def _write_single_constraint_fast(
    filename, eq_name, sense, lhs_terms, rhs_const, rhs_orig
):
    """Write single constraint using Rust - optimized for many variables."""
    all_var_names = []
    all_coefs = []

    for term in lhs_terms:
        var, coef, _ = term[0], term[1], term[2]
        if not var.sets:
            all_var_names.append(var.name)
            if isinstance(coef, (int, float)):
                c = float(coef)
            else:
                c = float(coef.values.flat[0])
            all_coefs.append(c)
            continue

        # Generate all variable names efficiently
        var_names = var.all_names()
        all_var_names.extend(var_names)

        # Coefficients aligned/broadcast to the variable's flat order
        all_coefs.extend(_coef_flat_for_var(coef, var).tolist())

    # Get RHS value
    if hasattr(rhs_orig, "array"):
        rhs = float(rhs_orig.values.flat[0])
    elif hasattr(rhs_orig, "values"):
        rhs = float(rhs_orig.values.flat[0])
    else:
        rhs = rhs_const

    nimopt_rust.write_single_constraint(
        filename,
        eq_name,
        sense,
        all_var_names,
        np.array(all_coefs, dtype=np.float64),
        rhs,
    )


def _write_single_constraint_py(
    filename, eq_name, sense, lhs_terms, rhs_const, bindings, rhs_orig
):
    """Write a single constraint using Python (fallback)."""
    with open(filename, "a") as f:
        f.write(f" {eq_name}: ")
        first = True
        for term in lhs_terms:
            var, coef, _ = term[0], term[1], term[2]
            var_combos = []
            for s in var.sets:
                if s in bindings:
                    var_combos.append([bindings[s]])
                else:
                    var_combos.append(s.elements)

            for combo in itertools.product(*var_combos) if var_combos else [()]:
                suffix = (
                    "_" + "_".join(sanitize_lp_name(e) for e in combo) if combo else ""
                )
                vname = var.name + suffix
                cv = _get_coef_scalar(coef, var.sets, combo)
                if cv == 0:
                    continue
                if first:
                    if cv == 1:
                        f.write(vname)
                    elif cv == -1:
                        f.write(f"- {vname}")
                    elif cv > 0:
                        f.write(f"{cv} {vname}")
                    else:
                        f.write(f"- {-cv} {vname}")
                    first = False
                else:
                    if cv == 1:
                        f.write(f" + {vname}")
                    elif cv == -1:
                        f.write(f" - {vname}")
                    elif cv > 0:
                        f.write(f" + {cv} {vname}")
                    else:
                        f.write(f" - {-cv} {vname}")

        if first:
            f.write("0")

        if isinstance(rhs_orig, nb.Array):
            rv = float(rhs_orig.values.flat[0])
        elif hasattr(rhs_orig, "array"):
            rv = float(rhs_orig.values.flat[0])
        elif hasattr(rhs_orig, "values"):
            rv = float(rhs_orig.values.flat[0])
        else:
            rv = rhs_const
        f.write(f" {sense} {rv}\n")


def _negate_coef(coef):
    if isinstance(coef, (int, float)):
        return -coef
    elif isinstance(coef, nb.Array):
        return coef * (-1)
    return coef


def _write_bounds_fast(model, filename):
    """Write bounds using Rust for variable name generation."""
    # Write header
    with open(filename, "a") as f:
        f.write("\nBounds\n")

    for var in model.variables.values():
        if var.sets:
            dim_elements = [[sanitize_lp_name(e) for e in s.elements] for s in var.sets]
            nimopt_rust.write_bounds_fast(
                filename, var.name, dim_elements, var.lb, var.ub
            )
        else:
            # Scalar variable - write directly
            with open(filename, "a") as f:
                if var.lb is None and var.ub is None:
                    f.write(f" {var.name} free\n")
                elif var.lb is None:
                    f.write(f" {var.name} <= {var.ub}\n")
                elif var.ub is None:
                    f.write(f" {var.name} >= {var.lb}\n")
                else:
                    f.write(f" {var.lb} <= {var.name} <= {var.ub}\n")


def _get_bounds_data(model):
    """Get all variable names and bounds."""
    all_vars = []
    lbs = []
    ubs = []
    for var in model.variables.values():
        for vn in var.all_names(sanitize=True):
            all_vars.append(vn)
            lbs.append(var.lb)
            ubs.append(var.ub)
    return all_vars, lbs, ubs


def _get_var_types(model):
    """Get integer and binary variable names."""
    ints = []
    bins = []
    for var in model.variables.values():
        if var.vtype == "integer":
            ints.extend(var.all_names(sanitize=True))
        elif var.vtype == "binary":
            bins.extend(var.all_names(sanitize=True))
    return ints, bins
