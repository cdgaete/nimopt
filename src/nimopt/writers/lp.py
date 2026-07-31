"""LP format writer."""

import itertools
from typing import Any, Dict, List, TextIO, Tuple

import nimblend as nb

from ..expression import LinearExpr
from ..sets import Set
from ..variable import Variable, VarRef
from . import sanitize_lp_name


def write_lp(model, filename: str) -> None:
    """Write model to LP format."""
    var_names = _build_var_names(model)
    with open(filename, "w", buffering=8 * 1024 * 1024) as f:
        f.write(f"\\ {model.name}\n\\ nimopt v2\n\n")
        _write_objective(f, model, var_names)
        _write_constraints(f, model, var_names)
        _write_bounds(f, model)
        _write_var_types(f, model)
        f.write("End\n")


def _build_var_names(model) -> Dict[Tuple, str]:
    """Pre-compute variable names: (var_id, indices) -> name."""
    names: Dict[Tuple, str] = {}
    for var in model.variables.values():
        if not var.sets:
            names[(id(var), ())] = var.name
        else:
            for combo in itertools.product(*(s.elements for s in var.sets)):
                vname = var.name + "_" + "_".join(sanitize_lp_name(e) for e in combo)
                names[(id(var), combo)] = vname
    return names


def _get_coef(
    coef, var_sets: List[Set], combo: tuple, bindings: Dict[Set, Any] = None
) -> float:
    """Get coefficient value for a specific variable index.

    Handles subset aliasing: when variable is indexed by a subset
    (e.g., Dispatchable) but coefficient is indexed by superset
    (e.g., Technologies), looks up by element value.

    If bindings is provided, also uses those for coefficient dimensions
    not covered by var_sets (e.g., Hours when var has only Tech).
    """
    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        indices = []
        used_dims = set()

        # Build full bindings: var_sets + outer bindings
        full_bindings = dict(bindings) if bindings else {}
        for s, elem in zip(var_sets, combo):
            full_bindings[s] = elem

        # Index into coefficient array using all bindings
        for dim in coef.dims:
            for s, elem in full_bindings.items():
                if s.name == dim and dim not in used_dims:
                    coord_list = list(coef.coords[dim])
                    try:
                        indices.append(coord_list.index(elem))
                        used_dims.add(dim)
                    except ValueError:
                        return 0.0
                    break
            # If dim not found in bindings, skip it

        if indices:
            return float(coef.values[tuple(indices)])
        return float(coef.values.flat[0])
    elif hasattr(coef, "array"):
        return _get_coef(coef.array, var_sets, combo, bindings)
    return float(coef)


def _index_array_by_bindings(arr, bindings: Dict[Set, Any]) -> float:
    """Value of ``arr`` at the coordinates named by ``bindings``.

    Indices are assembled in the array's OWN dimension order - not the order
    ``bindings`` happens to iterate. A name match wins; any remaining array
    dimension is resolved by locating the element in it (subset aliasing).
    Assembling indices in binding order silently transposed them whenever the
    constraint's free-set order differed from the array's dim order, indexing
    the wrong axis (and, when sizes differ, raising IndexError).
    """
    idx_by_dim: Dict[str, int] = {}

    # Exact dimension-name matches first.
    for dim in arr.dims:
        for s, elem in bindings.items():
            if s.name == dim:
                idx_by_dim[dim] = list(arr.coords[dim]).index(elem)
                break

    # Subset aliasing for still-unmatched dims: consume each binding at most once.
    used = {id(s) for s in bindings if s.name in idx_by_dim}
    for dim in arr.dims:
        if dim in idx_by_dim:
            continue
        coord_list = list(arr.coords[dim])
        for s, elem in bindings.items():
            if id(s) in used:
                continue
            if elem in coord_list:
                idx_by_dim[dim] = coord_list.index(elem)
                used.add(id(s))
                break

    indices = tuple(idx_by_dim[dim] for dim in arr.dims if dim in idx_by_dim)
    if len(indices) == len(arr.dims):
        return float(arr.values[indices])
    return float(arr.values.flat[0])


def _get_rhs(rhs, bindings: Dict[Set, Any]) -> float:
    """Get RHS value for constraint.

    Handles subset aliasing: looks up by element value when set names don't match.
    """
    if isinstance(rhs, (int, float)):
        return float(rhs)
    elif isinstance(rhs, nb.Array):
        return _index_array_by_bindings(rhs, bindings)
    elif hasattr(rhs, "array"):
        # Handle ParamRef - use its underlying array
        return _index_array_by_bindings(rhs.array, bindings)
    return 0.0


def _write_term(f: TextIO, vname: str, coef: float, first: bool) -> bool:
    """Write a single term to the file."""
    if coef == 0:
        return first
    if first:
        if coef == 1:
            f.write(vname)
        elif coef == -1:
            f.write(f"- {vname}")
        elif coef >= 0:
            f.write(f"{coef} {vname}")
        else:
            f.write(f"- {-coef} {vname}")
    else:
        if coef == 1:
            f.write(f" + {vname}")
        elif coef == -1:
            f.write(f" - {vname}")
        elif coef >= 0:
            f.write(f" + {coef} {vname}")
        else:
            f.write(f" - {-coef} {vname}")
    return False


def _write_objective(f: TextIO, model, var_names: Dict) -> None:
    """Write objective section."""
    f.write("Minimize\n" if model.sense == "minimize" else "Maximize\n")
    f.write(" obj: ")
    
    # Pre-accumulate coefficients for each variable to handle duplicates
    # HiGHS LP parser doesn't sum duplicate variable appearances
    coef_map: Dict[str, float] = {}
    
    if model.objective:
        for var, coef, fixed, _lagged in model.objective.terms:
            # Literal indices pin a dimension to their named element instead
            # of expanding it over its whole set.
            fixed_map = {pos: val for pos, val in fixed} if fixed else {}
            if var.sets:
                combos = list(
                    itertools.product(
                        *(
                            [fixed_map[i]] if i in fixed_map else s.elements
                            for i, s in enumerate(var.sets)
                        )
                    )
                )
            else:
                combos = [()]
            for combo in combos:
                vname = var_names[(id(var), combo)]
                cv = _get_coef(coef, var.sets, combo)
                coef_map[vname] = coef_map.get(vname, 0.0) + cv
    
    # Write accumulated coefficients
    first = True
    for vname, cv in coef_map.items():
        first = _write_term(f, vname, cv, first)
    
    if first:
        f.write("0")
    f.write("\n\n")


def _write_constraints(f: TextIO, model, var_names: Dict) -> None:
    """Write constraints section."""
    f.write("Subject To\n")
    sm = {"<=": "<=", ">=": ">=", "==": "="}

    for eq_name, con in model._constraints.items():
        # Normalize: move all terms to LHS, constant to RHS
        lhs_terms = list(con.lhs.terms)
        rhs_const = 0.0
        rhs_const_array = None  # Track array constant separately

        if isinstance(con.rhs, LinearExpr):
            for var, coef, fixed, lagged in con.rhs.terms:
                neg_coef = _negate_coef(coef)
                lhs_terms.append((var, neg_coef, fixed, lagged))
            if isinstance(con.rhs.const, (int, float)):
                # Un-negated: moving RHS var terms to the LHS leaves the RHS
                # constant on the RHS (`x == y + 5` -> `x - y = 5`). Matches the
                # direct solver build path.
                rhs_const = con.rhs.const
            elif isinstance(con.rhs.const, nb.Array):
                # Array const - handled separately during constraint iteration
                rhs_const_array = con.rhs.const
        elif isinstance(con.rhs, (Variable, VarRef)):
            var = con.rhs if isinstance(con.rhs, Variable) else con.rhs.var
            lhs_terms.append((var, -1.0, [], []))
        elif isinstance(con.rhs, (int, float)):
            rhs_const = float(con.rhs)

        norm_lhs = LinearExpr(lhs_terms, 0.0, con.lhs.free_sets)

        # A constant folded into the LHS expression (e.g. `z + 3 <= 9` or the
        # bare `ufix[A]` in `4*Sum(B, y[A,B]) + ufix[A] <= 1`) moves to the RHS
        # as RHS' = RHS - lhs.const. Track it as a separate offset so the
        # existing RHS sign handling below is left untouched; it is subtracted
        # per row (numeric part always, array part indexed by the row bindings).
        lhs_const = con.lhs.const
        lhs_off_num = 0.0
        lhs_off_arr = None
        if not (isinstance(lhs_const, (int, float)) and lhs_const == 0):
            if isinstance(lhs_const, (int, float)):
                lhs_off_num = float(lhs_const)
            else:
                lhs_off_arr = getattr(lhs_const, "array", lhs_const)

        free = con.free_sets
        if not free:
            f.write(f" {eq_name}: ")
            if not _write_lhs(f, norm_lhs, var_names, {}):
                # Lagged index out of bounds - remove partial write
                f.seek(f.tell() - len(f" {eq_name}: "))
                continue
            if rhs_const_array is not None:
                rv = -float(rhs_const_array.values.flat[0])
            elif isinstance(con.rhs, nb.Array):
                rv = _get_rhs(con.rhs, {})
            elif hasattr(con.rhs, "array"):
                rv = _get_rhs(con.rhs, {})
            else:
                rv = rhs_const
            rv -= lhs_off_num
            if lhs_off_arr is not None:
                rv -= float(lhs_off_arr.values.flat[0])
            f.write(f" {sm[con.sense]} {rv}\n")
        else:
            combos = list(itertools.product(*(s.elements for s in free)))
            con_idx = 0
            for combo in combos:
                bindings = dict(zip(free, combo))
                # Check if constraint should be skipped before writing name
                if not _check_lag_bounds(norm_lhs, bindings):
                    continue
                f.write(f" {eq_name}_{con_idx}: ")
                _write_lhs(f, norm_lhs, var_names, bindings)
                if rhs_const_array is not None:
                    rv = _get_rhs(rhs_const_array, bindings)
                elif isinstance(con.rhs, nb.Array):
                    rv = _get_rhs(con.rhs, bindings)
                elif hasattr(con.rhs, "array"):
                    rv = _get_rhs(con.rhs, bindings)
                else:
                    rv = rhs_const
                rv -= lhs_off_num
                if lhs_off_arr is not None:
                    rv -= _index_array_by_bindings(lhs_off_arr, bindings)
                f.write(f" {sm[con.sense]} {rv}\n")
                con_idx += 1
    f.write("\n")


def _negate_coef(coef):
    """Negate a coefficient."""
    if isinstance(coef, (int, float)):
        return -coef
    elif isinstance(coef, nb.Array):
        return coef * (-1)
    return coef


def _check_lag_bounds(expr: LinearExpr, bindings: Dict[Set, Any]) -> bool:
    """Check if any lagged indices are out of bounds.

    Returns True if constraint should be generated, False to skip.
    """
    for var, coef, fixed, lagged in expr.terms:
        if lagged:
            for pos, ls in lagged:
                base_set = ls.base_set
                if base_set in bindings:
                    elem = bindings[base_set]
                    lagged_elem = ls.get_lagged_element(elem)
                    if lagged_elem is None:
                        return False
    return True


def _write_lhs(
    f: TextIO, expr: LinearExpr, var_names: Dict, bindings: Dict[Set, Any]
) -> bool:
    """Write LHS expression terms.

    Returns False if constraint should be skipped (lagged index out of bounds).
    """

    # First pass: check if any lagged indices are out of bounds
    for var, coef, fixed, lagged in expr.terms:
        if lagged:
            for pos, ls in lagged:
                base_set = ls.base_set
                if base_set in bindings:
                    elem = bindings[base_set]
                    lagged_elem = ls.get_lagged_element(elem)
                    if lagged_elem is None:
                        return False  # Skip this constraint

    first = True
    for var, coef, fixed, lagged in expr.terms:
        # Build lagged_map for quick lookup
        lagged_map = {pos: ls for pos, ls in lagged} if lagged else {}
        # Honor fixed (literal) indices, e.g. vol[WR, 1]: pin that dimension
        # to the named element instead of expanding it over its whole set.
        fixed_map = {pos: val for pos, val in fixed} if fixed else {}

        var_combos = []
        for i, s in enumerate(var.sets):
            if i in fixed_map:
                var_combos.append([fixed_map[i]])
            elif s in bindings:
                var_combos.append([bindings[s]])
            elif i in lagged_map:
                # For lagged dims, use the binding for base set
                base_set = lagged_map[i].base_set
                if base_set in bindings:
                    var_combos.append([bindings[base_set]])
                else:
                    var_combos.append(base_set.elements)
            else:
                var_combos.append(s.elements)

        if var_combos:
            combos = list(itertools.product(*var_combos))
        else:
            combos = [()]

        for combo in combos:
            # Apply lag offset to get actual variable indices
            actual_combo = list(combo)
            for pos, ls in lagged_map.items():
                elem = combo[pos]
                lagged_elem = ls.get_lagged_element(elem)
                actual_combo[pos] = lagged_elem

            vname = var_names[(id(var), tuple(actual_combo))]
            cv = _get_coef(coef, var.sets, combo, bindings)
            first = _write_term(f, vname, cv, first)

    if first:
        f.write("0")

    return True


def _write_bounds(f: TextIO, model) -> None:
    """Write bounds section."""
    f.write("Bounds\n")
    for var in model.variables.values():
        for vn in var.all_names(sanitize=True):
            if var.lb is None and var.ub is None:
                f.write(f" {vn} free\n")
            elif var.lb is None:
                f.write(f" {vn} <= {var.ub}\n")
            elif var.ub is None:
                f.write(f" {vn} >= {var.lb}\n")
            else:
                f.write(f" {var.lb} <= {vn} <= {var.ub}\n")
    f.write("\n")


def _write_var_types(f: TextIO, model) -> None:
    """Write integer/binary variable sections."""
    ints = []
    bins = []
    for v in model.variables.values():
        if v.vtype == "integer":
            ints.extend(v.all_names(sanitize=True))
        elif v.vtype == "binary":
            bins.extend(v.all_names(sanitize=True))

    if ints:
        f.write("General\n")
        for n in ints:
            f.write(f" {n}\n")
        f.write("\n")

    if bins:
        f.write("Binary\n")
        for n in bins:
            f.write(f" {n}\n")
        f.write("\n")
