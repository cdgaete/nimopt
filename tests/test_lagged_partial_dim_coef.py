"""Regression tests: partial-dim coefficient on a LAGGED constraint.

Sibling of ``test_sum_partial_dim_coef.py``, which covers the ``Sum``
path. This file covers ``_build_lagged_constraint_vectorized``, the direct
solver's builder for any constraint carrying a lag/lead term.

In the canonical storage balance

    soc[S,T] == soc[S,T-1] + eff[S] * chg[S,T] - dis[S,T] / eff[S]   free [S,T]

the coefficient is indexed by ``S`` alone (|S| values) while the variable
spans ``[S,T]`` (|S|*|T| cells). The builder did two things wrong:

1. it flattened the coefficient array as-is, with no broadcast to the
   variable's dims (every other path -- both LP writers, the objective
   builder, the Sum path -- goes through ``_coef_flat_for_var``);
2. it then looked the coefficient up by the *constraint row* index and, when
   that ran past the end of the short array, silently fell back to
   ``coef[0]``.

Net effect: every row past ``|S|`` was assembled with the FIRST element's
coefficient. This is a wrong-answer bug, not a crash -- the model still
reports ``optimal``. It hides well in a fleet where most units share an
efficiency: only the minority units come out wrong. On PyPSA-CL's Chilean SEN
model it gave two 0.6-efficiency Carnot storage units the 0.9592 efficiency
of the first (BESS) unit, letting them discharge 281 MW for only 293 MWh of
stored energy instead of 469 -- free energy, and an over-build of that
technology.

The LP writers were correct throughout, so they serve as the oracle here:
every assertion is made against all four build paths.

Note the lag semantics being pinned: the coefficient is indexed by the
*un-lagged* combo even on the lagged term, matching the explicit comment in
``writers/lp_rust.py``'s lagged path.
"""

import numpy as np
import pytest

import nimopt as no
from nimopt import Sum

from test_sum_literal_coef import direct_rows, lp_rows

pytest.importorskip("nimopt_rust")
pytest.importorskip("highspy")

# Deliberately different per unit: with the bug, u2 was solved as if it had
# u1's efficiency, so any test that gave both the same value would pass.
EFF = {"u1": 0.8, "u2": 0.5}


def _model():
    S = no.Set("S", ["u1", "u2"])
    T = no.Set("T", ["t1", "t2"])
    eff = no.Param("eff", [S], [EFF["u1"], EFF["u2"]])
    m = no.Model("lagcoef", sense="minimize")
    soc = m.var("soc", [S, T], lb=0)
    chg = m.var("chg", [S, T], lb=0)
    dis = m.var("dis", [S, T], lb=0)
    m.eq("bal", soc[S, T] == soc[S, T.lag(1)] + eff[S] * chg[S, T] - dis[S, T] / eff[S])
    m.set_objective(Sum(S, T, chg[S, T] + soc[S, T]))
    return m


# The T=t1 row is dropped (non-cyclic lag out of bounds), leaving one row per
# unit. Each must carry ITS OWN efficiency, not u1's.
EXPECTED = {
    "u1": {
        "soc_u1_t2": 1.0, "soc_u1_t1": -1.0,
        "chg_u1_t2": -EFF["u1"], "dis_u1_t2": 1.0 / EFF["u1"],
    },
    "u2": {
        "soc_u2_t2": 1.0, "soc_u2_t1": -1.0,
        "chg_u2_t2": -EFF["u2"], "dis_u2_t2": 1.0 / EFF["u2"],
    },
}


def _by_unit(rows):
    """Row dicts keyed by unit. The direct builder names rows 'bal_u1_t2'
    while the LP writer names them 'bal_0', so match on content instead."""
    out = {}
    for _, payload in rows.items():
        row = payload[0]
        for unit in EFF:
            if f"soc_{unit}_t2" in row:
                out[unit] = row
    return out


def _assert_rows(rows):
    by_unit = _by_unit(rows)
    assert set(by_unit) == set(EFF), f"expected one row per unit, got {sorted(by_unit)}"
    for unit, expected in EXPECTED.items():
        row = by_unit[unit]
        for var, coef in expected.items():
            assert var in row, f"row for {unit} is missing term {var}: {row}"
            assert row[var] == pytest.approx(coef, rel=1e-12), (
                f"{unit}[{var}] = {row[var]}, expected {coef}"
            )


@pytest.mark.parametrize("use_rust", [True, False])
def test_direct_build_uses_each_units_own_coefficient(use_rust):
    _assert_rows(direct_rows(_model(), use_rust))


@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_export_uses_each_units_own_coefficient(tmp_path, use_rust):
    rows = {k: v for k, v in lp_rows(_model(), tmp_path, use_rust).items()}
    _assert_rows({k: (v[0], v[1], v[2]) for k, v in rows.items()})


def test_direct_and_lp_agree():
    """The LP path was always right; the direct path must match it exactly."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        lp = _by_unit(lp_rows(_model(), pathlib.Path(d), True))
    direct = _by_unit(direct_rows(_model(), True))
    for unit, expected in EXPECTED.items():
        for var in expected:
            assert direct[unit][var] == pytest.approx(lp[unit][var], rel=1e-12)


def test_solution_conserves_energy_per_unit():
    """End-to-end: the solved trajectory must satisfy the balance with each
    unit's real efficiency."""
    S = no.Set("S", [f"u{i}" for i in range(5)])
    T = no.Set("T", list(range(1, 5)))
    effs = [0.9592, 0.9, 0.77, 0.6, 0.5]
    eff = no.Param("eff", [S], effs)
    m = no.Model("lagsolve", sense="minimize")
    soc = m.var("soc", [S, T], lb=0)
    chg = m.var("chg", [S, T], lb=0)
    dis = m.var("dis", [S, T], lb=0)
    m.eq("bal", soc[S, T] == soc[S, T.lag(1)] + eff[S] * chg[S, T] - dis[S, T] / eff[S])
    m.eq("must", dis[S, T] >= 10)
    m.set_objective(Sum(S, T, chg[S, T] + soc[S, T]))
    _status, sol = m.solve()

    sv = np.asarray(sol.var("soc").values)
    cv = np.asarray(sol.var("chg").values)
    dv = np.asarray(sol.var("dis").values)
    worst = 0.0
    for u, e in enumerate(effs):
        for k in range(1, len(T)):
            worst = max(worst, abs(sv[u, k] - (sv[u, k - 1] + e * cv[u, k] - dv[u, k] / e)))
    assert worst < 1e-6, f"lagged balance violated by {worst}"
