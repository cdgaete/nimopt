"""Regression tests: coefficient indexed by a strict subset of the variable's dims.

In ``Sum(G, inertia[G] * status[G,T])`` free over ``[T]``, the coefficient is
indexed only by the *summed* set ``G`` while the variable spans ``[G, T]`` and
the free set ``T`` comes from the variable alone. The single-term Rust fast
path (``_build_sum_csr_rust_fast``) flattened the coefficient array as-is and
handed the kernel an array of length ``|G|`` where the kernel expects the
variable's full flat size ``|G|*|T|``. The kernel pads a short array with 1.0,
so only the *first* free-set element received the real coefficients; every
later row silently got ``1.0`` for most of its terms.

That is a wrong-answer bug, not a crash: with a ``>=`` requirement the padded
rows become unsatisfiable and the model reports INFEASIBLE (this is how it was
found -- a system-inertia constraint with 29x slack presolved to infeasible),
but with ``<=`` the model solves happily to a wrong optimum.

The multi-term builder and both LP writers broadcast correctly, which is why
only single-term ``Sum`` constraints routed to the fast path were affected.
"""

import numpy as np
import pytest

import nimopt as no
from nimopt import Sum

from test_sum_literal_coef import direct_rows, lp_rows

pytest.importorskip("nimopt_rust")
pytest.importorskip("highspy")


def _model(sense):
    """Constraint free over [T]:  Sum(G, inertia[G] * status[G,T]) <sense> 50.

    The coefficient ``inertia`` is indexed by G only; T is contributed solely
    by the variable.
    """
    G = no.Set("G", ["g1", "g2"])
    T = no.Set("T", ["t1", "t2"])
    inertia = no.Param("inertia", [G], [100.0, 5.0])
    m = no.Model("partial", sense="minimize")
    status = m.var("status", [G, T], lb=0, ub=1)
    expr = Sum(G, inertia[G] * status[G, T])
    con = {"<=": expr <= 50, ">=": expr >= 50, "==": expr == 50}[sense]
    m.eq("req", con)
    m.set_objective(Sum(G, T, status[G, T]))
    return m


# Every row must carry the real per-generator coefficients, not 1.0 padding.
EXPECTED_ROWS = {
    "req_t1": {"status_g1_t1": 100.0, "status_g2_t1": 5.0},
    "req_t2": {"status_g1_t2": 100.0, "status_g2_t2": 5.0},
}


def _bounds_for(sense, rhs):
    if sense == "<=":
        return (-np.inf, rhs)
    if sense == ">=":
        return (rhs, np.inf)
    return (rhs, rhs)


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
@pytest.mark.parametrize("use_rust", [True, False])
def test_direct_row_coefficients(sense, use_rust):
    rows = direct_rows(_model(sense), use_rust)
    for cname, exp_row in EXPECTED_ROWS.items():
        got_row, lower, upper = rows[cname]
        assert got_row == pytest.approx(exp_row), (
            f"{sense} rust={use_rust}: {cname} row {got_row} != {exp_row}"
        )
        exp_lo, exp_hi = _bounds_for(sense, 50.0)
        assert lower == pytest.approx(exp_lo)
        assert upper == pytest.approx(exp_hi)


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_row_coefficients(sense, use_rust, tmp_path):
    rows = lp_rows(_model(sense), tmp_path, use_rust)
    got_rows = [r for (r, _s, _rhs) in rows.values()]
    for exp_row in EXPECTED_ROWS.values():
        assert any(r == pytest.approx(exp_row) for r in got_rows), (
            f"{sense} LP rust={use_rust}: expected row {exp_row} not in {got_rows}"
        )


def test_ge_requirement_is_feasible():
    """Sum(G, inertia[G]*status[G,T]) >= 50 is satisfiable in every period.

    status[g1,t]=0.5 alone gives 50, so the minimum of sum(status) is 0.5 per
    period -> 1.0 overall. Under the padding bug the t2 row read
    ``1*status_g1_t2 + 1*status_g2_t2 >= 50`` with both vars capped at 1,
    making the model INFEASIBLE.
    """
    res, _sol = _model(">=").solve()
    assert res.status.value == "optimal"
    assert res.objective_value == pytest.approx(1.0)
