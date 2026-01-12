"""LP format writer."""

import itertools
from typing import Any, Dict, List, TextIO, Tuple

import nimblend as nb

from ..expression import LinearExpr
from ..sets import Set
from ..variable import Variable, VarRef


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
                vname = var.name + "_" + "_".join(str(e) for e in combo)
                names[(id(var), combo)] = vname
    return names


def _get_coef(coef, var_sets: List[Set], combo: tuple) -> float:
    """Get coefficient value for a specific variable index."""
    if isinstance(coef, (int, float)):
        return float(coef)
    elif isinstance(coef, nb.Array):
        indices = []
        for i, s in enumerate(var_sets):
            if s.name in coef.dims:
                elem = combo[i]
                coord_list = list(coef.coords[s.name])
                indices.append(coord_list.index(elem))
        if indices:
            return float(coef.values[tuple(indices)])
        return float(coef.values)
    elif hasattr(coef, "array"):
        return _get_coef(coef.array, var_sets, combo)
    return float(coef)


def _get_rhs(rhs, bindings: Dict[Set, Any]) -> float:
    """Get RHS value for constraint."""
    if isinstance(rhs, (int, float)):
        return float(rhs)
    elif isinstance(rhs, nb.Array):
        indices = []
        for s in bindings:
            if s.name in rhs.dims:
                coord_list = list(rhs.coords[s.name])
                indices.append(coord_list.index(bindings[s]))
        if indices:
            return float(rhs.values[tuple(indices)])
        return float(rhs.values.flat[0])
    elif hasattr(rhs, "array"):
        indices = [s.index(bindings[s]) for s in rhs.sets if s in bindings]
        if indices:
            return float(rhs.array.values[tuple(indices)])
        return float(rhs.values.flat[0])
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
    first = True
    if model.objective:
        for var, coef, fixed in model.objective.terms:
            if var.sets:
                combos = list(itertools.product(*(s.elements for s in var.sets)))
            else:
                combos = [()]
            for combo in combos:
                vname = var_names[(id(var), combo)]
                cv = _get_coef(coef, var.sets, combo)
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

        if isinstance(con.rhs, LinearExpr):
            for var, coef, fixed in con.rhs.terms:
                neg_coef = _negate_coef(coef)
                lhs_terms.append((var, neg_coef, fixed))
            if isinstance(con.rhs.const, (int, float)):
                rhs_const = -con.rhs.const
        elif isinstance(con.rhs, (Variable, VarRef)):
            var = con.rhs if isinstance(con.rhs, Variable) else con.rhs.var
            lhs_terms.append((var, -1.0, []))
        elif isinstance(con.rhs, (int, float)):
            rhs_const = float(con.rhs)

        norm_lhs = LinearExpr(lhs_terms, 0.0, con.lhs.free_sets)

        free = con.free_sets
        if not free:
            f.write(f" {eq_name}: ")
            _write_lhs(f, norm_lhs, var_names, {})
            if hasattr(con.rhs, "array"):
                rv = _get_rhs(con.rhs, {})
            else:
                rv = rhs_const
            f.write(f" {sm[con.sense]} {rv}\n")
        else:
            combos = list(itertools.product(*(s.elements for s in free)))
            for idx, combo in enumerate(combos):
                bindings = dict(zip(free, combo))
                f.write(f" {eq_name}_{idx}: ")
                _write_lhs(f, norm_lhs, var_names, bindings)
                if hasattr(con.rhs, "array"):
                    rv = _get_rhs(con.rhs, bindings)
                else:
                    rv = rhs_const
                f.write(f" {sm[con.sense]} {rv}\n")
    f.write("\n")


def _negate_coef(coef):
    """Negate a coefficient."""
    if isinstance(coef, (int, float)):
        return -coef
    elif isinstance(coef, nb.Array):
        return coef * (-1)
    return coef


def _write_lhs(
    f: TextIO, expr: LinearExpr, var_names: Dict, bindings: Dict[Set, Any]
) -> None:
    """Write LHS expression terms."""
    first = True
    for var, coef, fixed in expr.terms:
        var_combos = []
        for s in var.sets:
            if s in bindings:
                var_combos.append([bindings[s]])
            else:
                var_combos.append(s.elements)

        if var_combos:
            combos = list(itertools.product(*var_combos))
        else:
            combos = [()]

        for combo in combos:
            vname = var_names[(id(var), combo)]
            cv = _get_coef(coef, var.sets, combo)
            first = _write_term(f, vname, cv, first)

    if first:
        f.write("0")


def _write_bounds(f: TextIO, model) -> None:
    """Write bounds section."""
    f.write("Bounds\n")
    for var in model.variables.values():
        for vn in var.all_names():
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
            ints.extend(v.all_names())
        elif v.vtype == "binary":
            bins.extend(v.all_names())

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
