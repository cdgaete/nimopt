"""LP writer using Rust acceleration with optimized paths."""

import itertools
from typing import List

import nimblend as nb
import numpy as np

try:
    import nimopt_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from ..sets import Set


def write_lp_rust(model, filename: str) -> None:
    """Write LP using Rust-accelerated functions."""
    if not HAS_RUST:
        from .lp import write_lp

        return write_lp(model, filename)

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

    for term in model.objective.terms:
        var, coef, _ = term[0], term[1], term[2]
        if var.sets:
            dim_elements = [[str(e) for e in s.elements] for s in var.sets]
            if isinstance(coef, nb.Array):
                coefs = coef.values.flatten().astype(np.float64)
            elif hasattr(coef, "array"):
                coefs = coef.array.values.flatten().astype(np.float64)
            elif isinstance(coef, (int, float)):
                size = 1
                for s in var.sets:
                    size *= len(s)
                coefs = np.full(size, float(coef), dtype=np.float64)
            else:
                size = 1
                for s in var.sets:
                    size *= len(s)
                coefs = np.full(size, float(coef), dtype=np.float64)
            nimopt_rust.write_objective_fast(filename, var.name, dim_elements, coefs)
        else:
            # Scalar variable - use old method
            if isinstance(coef, (int, float)):
                c = float(coef)
            elif hasattr(coef, "values"):
                c = float(coef.values.flat[0])
            else:
                c = float(coef)
            nimopt_rust.write_objective(filename, [var.name], np.array([c]))


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
                    suffix = "_" + "_".join(str(e) for e in combo)
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
    var_dim_elements = [[str(e) for e in s.elements] for s in var.sets]
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

    # RHS array - need to compute in free set order
    n_free = 1
    for s in free_sets:
        n_free *= len(s)

    if hasattr(rhs_orig, "array"):
        rhs_flat = rhs_orig.array.values.flatten().astype(np.float64)
    elif hasattr(rhs_orig, "values"):
        rhs_flat = rhs_orig.values.flatten().astype(np.float64)
    else:
        rhs_flat = np.full(n_free, rhs_const, dtype=np.float64)

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
    """Write a batch of constraints using Rust (fallback path)."""
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
                suffix = "_" + "_".join(str(e) for e in combo) if combo else ""
                vname = var.name + suffix
                cv = _get_coef_scalar(coef, var.sets, combo)
                con_var_names.append(vname)
                con_coefs.append(cv)

        if free_idx == 0:
            vars_per_con = len(con_var_names)

        all_var_names.extend(con_var_names)
        all_coefs.extend(con_coefs)

        if hasattr(rhs_orig, "array"):
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

        # Get coefficients - vectorized if possible
        if isinstance(coef, (int, float)):
            all_coefs.extend([float(coef)] * len(var_names))
        elif isinstance(coef, nb.Array):
            # Flatten in the same order as itertools.product
            all_coefs.extend(coef.values.flatten().tolist())
        elif hasattr(coef, "array"):
            all_coefs.extend(coef.array.values.flatten().tolist())
        else:
            all_coefs.extend([float(coef)] * len(var_names))

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
                suffix = "_" + "_".join(str(e) for e in combo) if combo else ""
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

        if hasattr(rhs_orig, "array"):
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
            dim_elements = [[str(e) for e in s.elements] for s in var.sets]
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
        for vn in var.all_names():
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
            ints.extend(var.all_names())
        elif var.vtype == "binary":
            bins.extend(var.all_names())
    return ints, bins
