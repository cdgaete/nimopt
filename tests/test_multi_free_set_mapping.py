"""Regression: multi-term constraints over 2+ free sets with a term that is
free over only a *non-aligned subset* of those sets.

The Rust builders (``build_multi_term_csr`` for solve, ``write_multi_term_
constraints`` for LP export) used to map a term's free dimensions to the
constraint's free sets by a *positional counter*, assuming every term is free
over all the free sets in order. A term free over only the second set (e.g.
``x[B]`` in a ``[A, B]`` balance) got the first free set's index instead - which
silently corrupted the matrix (feasible models came back INFEASIBLE) and, when
the misused index exceeded a dimension's length, panicked out of bounds inside
the LP writer (surfacing as an uncaught ``PanicException`` that wedged the
server). The coefficient side is also indexed over summed axes so an incidence
coefficient ``a[i, j]`` no longer collapses to ``a[i, 0]``.
"""

import numpy as np
import pytest

import nimopt as no

pytest.importorskip("nimopt_rust")
pytest.importorskip("highspy")


def _partial_free_model():
    """min sum(y) + sum(x) s.t. y[a, b] + x[b] == d[a, b], y, x >= 0.

    x[b] is shared across a, so x[b] <= min_a d[a, b]; the optimum drives
    x[b] = min_a d[a, b] and gives objective sum(d) - sum_b min_a d[a, b].
    With d = [[1, 2, 3], [4, 5, 6]] that is 21 - (1 + 2 + 3) = 15.
    """
    m = no.Model("partial_free", sense="minimize")
    A = no.Set("A", ["a1", "a2"])
    B = no.Set("B", ["b1", "b2", "b3"])
    d = no.Param("d", [A, B], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = m.var("y", sets=[A, B], lb=0)
    x = m.var("x", sets=[B], lb=0)  # free over B only -> the non-aligned term
    m.eq("bal", y[A, B] + x[B] == d[A, B])
    m.set_objective(no.Sum(A, B, y[A, B]) + no.Sum(B, x[B]))
    return m


def test_direct_solve_partial_free_term_feasible():
    """Direct solver: a trivially feasible model must not come back infeasible."""
    m = _partial_free_model()
    result, sol = m.solve()
    assert "optimal" in str(result.status).lower(), result.status
    assert abs(result.objective_value - 15.0) < 1e-6
    xv = sol.variables["x"]
    bi = {b: i for i, b in enumerate(xv.elements[0])}
    assert np.allclose([xv.values[bi[b]] for b in ("b1", "b2", "b3")], [1.0, 2.0, 3.0])


def test_lp_writers_agree_partial_free_term(tmp_path):
    """Rust and Python LP writers must both export a solvable, correct LP
    (the Rust path used to panic / freeze and emit mis-indexed rows)."""
    import highspy

    m = _partial_free_model()
    objs = {}
    for label, use_rust in (("rust", True), ("python", False)):
        f = str(tmp_path / f"{label}.lp")
        m.to_lp(f, use_rust=use_rust)  # must not raise / hang
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(f)
        h.run()
        assert "Optimal" in str(h.getModelStatus()), f"{label} LP not optimal"
        objs[label] = h.getInfo().objective_function_value
    assert abs(objs["rust"] - objs["python"]) < 1e-9
    assert abs(objs["rust"] - 15.0) < 1e-6


def _virtual_free_set_model():
    """A single-term Sum reduced to a free set the variable is NOT indexed by:
    per-group emission caps ``Sum(g, e[g] * p[g]) <= cap[grp]`` free over GRP.
    p is indexed by G only; GRP lives solely in the coefficient e_grp[g, grp].
    """
    m = no.Model("virtual_free", sense="maximize")
    G = no.Set("G", ["g1", "g2", "g3"])
    GRP = no.Set("GRP", ["north", "south"])
    # g1,g2 in north ; g3 in south (membership carried in the coefficient)
    member = no.Param("member", [G, GRP], np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    cap = no.Param("cap", [GRP], np.array([10.0, 4.0]))
    p = m.var("p", sets=[G], lb=0, ub=100)
    m.eq("grp_cap", no.Sum(G, member[G, GRP] * p[G]) <= cap[GRP])
    m.set_objective(no.Sum(G, p[G]))
    return m


def test_virtual_free_set_grouping():
    """Free set present only in the coefficient must yield one row per group
    with the right membership (north cap 10 over g1+g2, south cap 4 over g3)."""
    m = _virtual_free_set_model()
    result, sol = m.solve()
    assert "optimal" in str(result.status).lower(), result.status
    # north: p_g1 + p_g2 <= 10 ; south: p_g3 <= 4 ; maximize sum -> 14
    assert abs(result.objective_value - 14.0) < 1e-6


def _block_neutral_model(rhs_kind):
    """min sum(price*dev) s.t. per-node, per-block  sum_t blk[b,t]*dev[n,t] == 0,
    dev in [-1, 1].

    BLK lives ONLY in the coefficient blk[BLK, T]; the variable dev is indexed
    by [NODE, T]. NODE is the variable's own free set, so BLK is an *additional*
    free set carried purely by the coefficient. With a constant ``== 0`` RHS,
    nothing on the RHS reintroduces BLK, so the constraint's free-set collection
    must recover it from the coefficient itself.

    rhs_kind="const"  -> ``== 0``            (the case that used to drop BLK)
    rhs_kind="param"  -> ``== zeroB[BLK]``   (RHS rescues BLK; proven correct)

    Both are the SAME math, so they must solve to the same objective. Under the
    bug the ``const`` model builds only 2 of the 4 neutrality rows (each node
    gets exactly one block), leaving half the deviations unconstrained -> a
    strictly lower (wrong) objective.
    """
    m = no.Model(f"blk_{rhs_kind}", sense="minimize")
    NODE = no.Set("NODE", ["n0", "n1"])
    T = no.Set("T", [1, 2, 3, 4])
    BLK = no.Set("BLK", ["b1", "b2"])
    # b1 -> t in {1,2}, b2 -> t in {3,4}
    blk = no.Param("blk", [BLK, T], np.array([[1.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 1.0]]))
    # distinct prices so the optimum is a nontrivial +1/-1 split within each block
    price = no.Param("price", [NODE, T], np.array([[3.0, 1.0, 5.0, 2.0],
                                                   [4.0, 6.0, 1.0, 8.0]]))
    dev = m.var("dev", sets=[NODE, T], lb=-1.0, ub=1.0)
    if rhs_kind == "const":
        m.eq("neutral", no.Sum(T, blk[BLK, T] * dev[NODE, T]) == 0)
    else:
        zeroB = no.Param("zeroB", [BLK], np.zeros(2))
        m.eq("neutral", no.Sum(T, blk[BLK, T] * dev[NODE, T]) == zeroB[BLK])
    m.set_objective(no.Sum(NODE, T, price[NODE, T] * dev[NODE, T]))
    return m


def test_coef_only_free_set_const_rhs_matches_param_rhs():
    """A free set carried only by an LHS coefficient must be recovered even when
    the RHS is a bare constant (== 0), not just when the RHS mentions it."""
    m_const = _block_neutral_model("const")
    m_param = _block_neutral_model("param")
    r_const, _ = m_const.solve()
    r_param, _ = m_param.solve()
    assert "optimal" in str(r_const.status).lower(), r_const.status
    assert "optimal" in str(r_param.status).lower(), r_param.status
    # identical math -> identical optimum; the bug makes const strictly lower
    assert abs(r_const.objective_value - r_param.objective_value) < 1e-6, (
        f"const={r_const.objective_value} param={r_param.objective_value}"
    )


def test_coef_only_free_set_const_rhs_row_count(tmp_path):
    """The == 0 form must emit one row per (NODE, BLK) = 4 rows, both writers."""
    for use_rust in (True, False):
        m = _block_neutral_model("const")
        f = str(tmp_path / f"blk_{int(use_rust)}.lp")
        m.to_lp(f, use_rust=use_rust)
        rows = [ln for ln in open(f) if ln.strip().startswith("neutral")]
        assert len(rows) == 4, f"use_rust={use_rust}: got {len(rows)} rows:\n{''.join(rows)}"


def _fused_coef_model(fused, rhs_kind):
    """Water-balance shape: ``Sum(WL, inc[WL, WN] * eff[WL] * f[WL, T]) == 0``.

    WN is carried ONLY by the coefficient; T is the variable's own free set.
    That is the same shape as ``_block_neutral_model`` with one difference: the
    coefficient is the *product of two params*, which ``ParamRef.__mul__``
    evaluates eagerly to a bare ``nb.Array`` before it ever reaches
    ``LinearExpr.from_term``.

    ``from_term`` only collects ``coef_sets`` for ``Param``/``ParamRef``
    coefficients; the ``nb.Array`` branch leaves it ``None``, so the free-set
    recovery loop is skipped and WN is dropped -- exactly the failure the
    single-param tests above were written to prevent, one branch further down.

    fused=False -> ``inc[WL, WN] * f[WL, T]``            (single ParamRef coef)
    fused=True  -> ``inc[WL, WN] * eff[WL] * f[WL, T]``  (nb.Array coef)

    Both forms are the same math (eff only rescales columns), and with a param
    RHS both build correctly; only (fused, const-RHS) loses rows.
    """
    m = no.Model(f"fused_{fused}_{rhs_kind}", sense="minimize")
    WL = no.Set("WL", ["l1", "l2", "l3"])
    WN = no.Set("WN", ["n1", "n2"])
    T = no.Set("T", [1, 2])
    inc = no.Param("inc", [WL, WN], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
    eff = no.Param("eff", [WL], np.array([1.0, 1.0, 1.0]))  # identity: math unchanged
    f = m.var("f", sets=[WL, T], lb=0, ub=1)
    coef_term = inc[WL, WN] * eff[WL] * f[WL, T] if fused else inc[WL, WN] * f[WL, T]
    if rhs_kind == "const":
        m.eq("bal", no.Sum(WL, coef_term) == 0)
    else:
        zero = no.Param("zero", [WN, T], np.zeros((2, 2)))
        m.eq("bal", no.Sum(WL, coef_term) == zero[WN, T])
    # distinct prices so every f is pinned; unconstrained rows show up as a
    # strictly lower objective rather than a tie.
    price = no.Param("price", [WL, T], np.array([[3.0, 1.0], [5.0, 2.0], [4.0, 6.0]]))
    m.set_objective(no.Sum(WL, T, price[WL, T] * f[WL, T]))
    return m


@pytest.mark.parametrize("use_rust", [True, False])
def test_fused_param_coef_const_rhs_row_count(tmp_path, use_rust):
    """A param*param coefficient must not lose the coefficient-only free set.

    Expect |WN| * |T| = 4 rows. Under the bug the fused form emits 2, zipping
    WN against T diagonally instead of taking the cross product.
    """
    m = _fused_coef_model(fused=True, rhs_kind="const")
    f = str(tmp_path / f"fused_{int(use_rust)}.lp")
    m.to_lp(f, use_rust=use_rust)
    rows = [ln for ln in open(f) if ln.strip().startswith("bal")]
    assert len(rows) == 4, f"use_rust={use_rust}: got {len(rows)} rows:\n{''.join(rows)}"


def test_unbindable_coef_dimension_raises():
    """A coefficient dimension nothing can bind must fail loudly at eq() time.

    A raw nimblend Array carries dimension names but no Set objects, so WN
    cannot reach the constraint's free sets by any route. Previously the
    writers skipped such a dimension and read an arbitrary slice; the model
    built and solved while meaning something other than what was written.
    """
    m = no.Model("unbindable", sense="minimize")
    WL = no.Set("WL", ["l1", "l2", "l3"])
    WN = no.Set("WN", ["n1", "n2"])
    T = no.Set("T", [1, 2])
    inc = no.Param("inc", [WL, WN], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
    f = m.var("f", sets=[WL, T], lb=0)
    with pytest.raises(ValueError, match="cannot be bound"):
        m.eq("bal", no.Sum(WL, inc.array * f[WL, T]) == 0)


def test_guard_allows_subset_aliasing():
    """The guard must not fire on a legitimate subset-aliased coefficient.

    ``cost`` is indexed by the superset ALL while the variable is indexed by
    the subset SUB; the writers bind that by element membership, not by name,
    so this must build.
    """
    m = no.Model("aliasing", sense="minimize")
    ALL = no.Set("ALL", ["a", "b", "c"])
    SUB = no.Set("SUB", ["a", "b"])
    T = no.Set("T", [1, 2])
    cost = no.Param("cost", [ALL], np.array([1.0, 2.0, 3.0]))
    x = m.var("x", sets=[SUB, T], lb=0)
    m.eq("cap", cost[ALL] * x[SUB, T] <= 5.0)  # must not raise
    assert "cap" in m._constraints


def test_fused_param_coef_matches_single_param_coef():
    """eff is all ones, so fusing it in must not change the optimum.

    This is the silent-wrong-answer face of the bug: the fused model stays
    feasible and reports a strictly better objective because half its balance
    rows were never built.
    """
    r_single, _ = _fused_coef_model(fused=False, rhs_kind="const").solve()
    r_fused, _ = _fused_coef_model(fused=True, rhs_kind="const").solve()
    assert "optimal" in str(r_single.status).lower(), r_single.status
    assert "optimal" in str(r_fused.status).lower(), r_fused.status
    assert abs(r_fused.objective_value - r_single.objective_value) < 1e-6, (
        f"single={r_single.objective_value} fused={r_fused.objective_value}"
    )
