"""Regression tests: literal (numeric) coefficient on a single-term constraint.

A literal coefficient multiplying a variable -- whether inside a ``Sum`` or on a
plain indexed term -- was silently dropped by the single-term Rust fast path
(``_build_sum_csr_rust_fast`` for direct solves, ``_write_sum_constraints_fast``
for LP export). Both handed the Rust kernel ``coef_flat=None`` for any scalar
coefficient, and the kernel reads ``None`` as "all coefficients are 1.0", so a
``4 * y`` term was assembled as ``1 * y``. The bug is independent of the
comparison operator: it fires for ``==``, ``<=`` and ``>=`` alike, whenever the
constraint routes to the single-term fast path (one variable, free sets equal
to the variable's own free dims). It only *looked* operator-specific because a
``z == Sum(...)`` control is multi-term (the ``z`` term forces the multi-term
builder, which materialises scalars correctly).

A second, distinct defect lives on the RHS-normalisation side: when a
constraint carries BOTH a numeric RHS constant and a folded LHS/RHS constant
*array* (e.g. ``4*Sum(B, y[A,B]) + ufix[A] <= 1``), the scalar constant was
dropped and only the array survived. See ``test_sum_plus_bare_param_rhs``.

Every test asserts on the assembled row coefficients / RHS directly (not just
the objective) across all four build paths: direct-solve Rust, direct-solve
pure-Python, LP-export Rust, LP-export pure-Python.
"""

import re

import numpy as np
import pytest

import nimopt as no
from nimopt import Sum
from nimopt.solvers.highs_direct import (
    _build_matrices,
    _build_matrices_rust_fast,
    _generate_var_names_from_info,
)

pytest.importorskip("nimopt_rust")
pytest.importorskip("highspy")


# --------------------------------------------------------------------------
# Row extraction helpers -- return {con_name: (row_dict, lower, upper)} where
# row_dict maps solver variable name -> coefficient.
# --------------------------------------------------------------------------


def _rows_from_matrices(mat, var_names):
    indptr = mat["indptr"]
    indices = mat["indices"]
    data = mat["data"]
    con_names = mat["con_names"]
    row_lower = mat["row_lower"]
    row_upper = mat["row_upper"]
    out = {}
    for i, name in enumerate(con_names):
        row = {}
        for k in range(indptr[i], indptr[i + 1]):
            row[var_names[indices[k]]] = float(data[k])
        out[name] = (row, float(row_lower[i]), float(row_upper[i]))
    return out


def direct_rows(m, use_rust):
    if use_rust:
        mat = _build_matrices_rust_fast(m)
        var_names = _generate_var_names_from_info(mat["var_info"])
    else:
        mat = _build_matrices(m)
        var_names = mat["var_names"]
    return _rows_from_matrices(mat, var_names)


# Leading sign is optional (the first term of a row has none).
_TERM_RE = re.compile(r"([+-]?)\s*([0-9.eE]*)\s*([A-Za-z_][A-Za-z0-9_]*)")


def lp_rows(m, tmp_path, use_rust):
    f = str(tmp_path / ("rust.lp" if use_rust else "py.lp"))
    m.to_lp(f, use_rust=use_rust)
    text = open(f).read()
    body = text.split("Subject To", 1)[1].split("Bounds", 1)[0]
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        mm = re.search(r"(<=|>=|=)\s*(-?[0-9.eE]+)\s*$", rest)
        assert mm, f"could not parse sense/rhs from {rest!r}"
        sense, rhs = mm.group(1), float(mm.group(2))
        lhs = rest[: mm.start()]
        row = {}
        for sign, num, var in _TERM_RE.findall(lhs):
            num = num.strip()
            coef = float(num) if num not in ("", ".") else 1.0
            if sign == "-":
                coef = -coef
            row[var] = row.get(var, 0.0) + coef
        out[name.strip()] = (row, sense, rhs)
    return out


# --------------------------------------------------------------------------
# Fixtures for the full coefficient matrix.
# --------------------------------------------------------------------------

SENSES = ["==", "<=", ">="]


def _model_sum(coef_kind, sense):
    """Constraint free over [A]:  Sum(B, COEF * y[A,B]) <sense> h[A]."""
    A = no.Set("A", ["a1", "a2"])
    B = no.Set("B", ["b1", "b2"])
    h = no.Param("h", [A], [0.5, 0.7])
    m = no.Model("sum", sense="maximize")
    y = m.var("y", [A, B], lb=0, ub=1)
    if coef_kind == "literal":
        expr = Sum(B, 4 * y[A, B])
        expected = {  # con_name -> {var: coef}
            "cap_a1": {"y_a1_b1": 4.0, "y_a1_b2": 4.0},
            "cap_a2": {"y_a2_b1": 4.0, "y_a2_b2": 4.0},
        }
    else:
        cost = no.Param("cost", [A, B], [[2.0, 3.0], [5.0, 7.0]])
        expr = Sum(B, cost[A, B] * y[A, B])
        expected = {
            "cap_a1": {"y_a1_b1": 2.0, "y_a1_b2": 3.0},
            "cap_a2": {"y_a2_b1": 5.0, "y_a2_b2": 7.0},
        }
    con = {"==": expr == h[A], "<=": expr <= h[A], ">=": expr >= h[A]}[sense]
    m.eq("cap", con)
    m.set_objective(Sum(A, B, y[A, B]))
    rhs = {"a1": 0.5, "a2": 0.7}
    return m, expected, rhs


def _model_plain(coef_kind, sense):
    """Constraint free over [A]:  COEF * w[A] <sense> h[A]."""
    A = no.Set("A", ["a1", "a2"])
    h = no.Param("h", [A], [0.5, 0.7])
    m = no.Model("plain", sense="maximize")
    w = m.var("w", [A], lb=0, ub=1)
    if coef_kind == "literal":
        expr = 4 * w[A]
        expected = {"cap_a1": {"w_a1": 4.0}, "cap_a2": {"w_a2": 4.0}}
    else:
        g = no.Param("g", [A], [2.0, 3.0])
        expr = g[A] * w[A]
        expected = {"cap_a1": {"w_a1": 2.0}, "cap_a2": {"w_a2": 3.0}}
    con = {"==": expr == h[A], "<=": expr <= h[A], ">=": expr >= h[A]}[sense]
    m.eq("cap", con)
    m.set_objective(Sum(A, w[A]))
    rhs = {"a1": 0.5, "a2": 0.7}
    return m, expected, rhs


def _bounds_for(sense, rhs):
    if sense == "<=":
        return (-np.inf, rhs)
    if sense == ">=":
        return (rhs, np.inf)
    return (rhs, rhs)


@pytest.mark.parametrize("shape", ["sum", "plain"])
@pytest.mark.parametrize("coef_kind", ["literal", "param"])
@pytest.mark.parametrize("sense", SENSES)
def test_direct_row_coefficients(shape, coef_kind, sense):
    build = _model_sum if shape == "sum" else _model_plain
    for use_rust in (True, False):
        m, expected, rhs = build(coef_kind, sense)
        rows = direct_rows(m, use_rust)
        for cname, exp_row in expected.items():
            got_row, lower, upper = rows[cname]
            assert got_row == pytest.approx(exp_row), (
                f"{shape}/{coef_kind}/{sense} rust={use_rust}: "
                f"{cname} row {got_row} != {exp_row}"
            )
            suffix = cname.split("_", 1)[1]
            exp_lo, exp_hi = _bounds_for(sense, rhs[suffix])
            assert lower == pytest.approx(exp_lo)
            assert upper == pytest.approx(exp_hi)


@pytest.mark.parametrize("shape", ["sum", "plain"])
@pytest.mark.parametrize("coef_kind", ["literal", "param"])
@pytest.mark.parametrize("sense", SENSES)
def test_lp_row_coefficients(shape, coef_kind, sense, tmp_path):
    build = _model_sum if shape == "sum" else _model_plain
    for use_rust in (True, False):
        m, expected, rhs = build(coef_kind, sense)
        rows = lp_rows(m, tmp_path, use_rust)
        # Constraint-row naming differs between writers (element vs positional),
        # so compare the multiset of coefficient rows rather than keying by name.
        got_rows = [r for (r, _s, _rhs) in rows.values()]
        for exp_row in expected.values():
            assert any(r == pytest.approx(exp_row) for r in got_rows), (
                f"{shape}/{coef_kind}/{sense} LP rust={use_rust}: "
                f"expected row {exp_row} not found among {got_rows}"
            )


def test_sum_literal_coef_objective():
    """End-to-end: Sum(B, 4*y[A,B]) <= h[A] maximise sum(y) -> 0.3, not 1.2."""
    m, _, _ = _model_sum("literal", "<=")
    res, sol = m.solve()
    assert res.status.value == "optimal"
    assert res.objective_value == pytest.approx(0.3)


# --------------------------------------------------------------------------
# Distinct RHS-normalisation bug: numeric RHS const dropped when a folded
# LHS/RHS constant array is also present.
# --------------------------------------------------------------------------


def test_sum_plus_bare_param_rhs(tmp_path):
    """4*Sum(B, y[A,B]) + ufix[A] <= 1  =>  4*Sum <= 1 - ufix[A].

    Coefficients (4.0) come from the multi-term builder and were always right;
    the defect was the RHS: the numeric ``1`` was dropped, leaving ``<= -ufix``.
    """
    def build():
        A = no.Set("A", ["a1", "a2"])
        B = no.Set("B", ["b1", "b2"])
        ufix = no.Param("ufix", [A], [0.2, 0.5])
        m = no.Model("mix", sense="maximize")
        y = m.var("y", [A, B], lb=0, ub=1)
        m.eq("mix", 4 * Sum(B, y[A, B]) + ufix[A] <= 1)
        m.set_objective(Sum(A, B, y[A, B]))
        return m

    expected_rows = {
        "mix_a1": {"y_a1_b1": 4.0, "y_a1_b2": 4.0},
        "mix_a2": {"y_a2_b1": 4.0, "y_a2_b2": 4.0},
    }
    expected_upper = {"mix_a1": 0.8, "mix_a2": 0.5}  # 1 - ufix

    for use_rust in (True, False):
        rows = direct_rows(build(), use_rust)
        for cname, exp_row in expected_rows.items():
            got_row, lower, upper = rows[cname]
            assert got_row == pytest.approx(exp_row), f"rust={use_rust} {cname}"
            assert upper == pytest.approx(expected_upper[cname]), (
                f"rust={use_rust} {cname} upper={upper} != {expected_upper[cname]}"
            )

    # LP-export path (both writers): coefficients 4.0 and RHS = 1 - ufix.
    exp_upper_vals = sorted(expected_upper.values())
    for use_rust in (True, False):
        rows = lp_rows(build(), tmp_path, use_rust)
        got_rows = [r for (r, _s, _rhs) in rows.values()]
        for exp_row in expected_rows.values():
            assert any(r == pytest.approx(exp_row) for r in got_rows), (
                f"LP rust={use_rust}: {exp_row} not in {got_rows}"
            )
        got_upper = sorted(rhs for (_r, s, rhs) in rows.values() if s == "<=")
        assert got_upper == pytest.approx(exp_upper_vals), (
            f"LP rust={use_rust}: RHS {got_upper} != {exp_upper_vals}"
        )

    res, sol = build().solve()
    assert res.status.value == "optimal"
    # a1: 4*sum <= 0.8 -> sum y = 0.2 ; a2: 4*sum <= 0.5 -> sum y = 0.125
    assert res.objective_value == pytest.approx(0.325)
