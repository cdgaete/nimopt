"""Regression tests: Rust LP writer coefficient broadcasting and
LP-path solution name alignment.

The Rust writer's scalar-constraint route used to flatten a
partial-dimension coefficient (e.g. ef[G] against p[G,T]) without
broadcasting, silently corrupting the constraint. LP-loaded solutions
also used to be sliced positionally although LP column order follows
first textual appearance, not model insertion order.
"""

import numpy as np
import pytest

import nimopt as no

pytest.importorskip("nimopt_rust")
pytest.importorskip("highspy")


def _dispatch_model():
    """2 gens x 3 hours; solar has zero cost (forces LP column reorder)
    and zero emissions; co2 cap is a scalar Sum constraint with a
    G-indexed coefficient against a (G,T) variable."""
    m = no.Model("reg", sense="minimize")
    G = no.Set("G", ["solar", "coal"])
    T = no.Set("T", ["t0", "t1", "t2"])
    mc = no.Param("mc", [G], np.array([0.0, 50.0]))
    ef = no.Param("ef", [G], np.array([0.0, 1.0]))
    avail = no.Param("avail", [G, T], np.array([[5.0, 5.0, 0.0], [10.0, 10.0, 10.0]]))
    dem = no.Param("demand", [T], np.array([8.0, 8.0, 8.0]))
    p = m.var("p", sets=[G, T], lb=0)
    m.eq("balance", no.Sum(G, p[G, T]) == dem[T])
    m.eq("max_output", p[G, T] <= avail[G, T])
    m.eq("co2_cap", no.Sum(G, T, ef[G] * p[G, T]) <= 15.0)
    m.set_objective(no.Sum(G, T, mc[G] * p[G, T]))
    return m


def test_rust_lp_scalar_constraint_broadcasts_coef(tmp_path):
    """Rust and Python writers must produce equivalent LPs when a scalar
    Sum constraint carries a coefficient over a subset of the variable's
    dims. Optimum: coal covers all residual demand; cap 15 binds at t2
    only via availability, cost = 50 * (3+3+8) = 700."""
    import highspy

    m = _dispatch_model()
    objs = {}
    for label, use_rust in (("rust", True), ("python", False)):
        f = str(tmp_path / f"{label}.lp")
        m.to_lp(f, use_rust=use_rust)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(f)
        h.run()
        assert "Optimal" in str(h.getModelStatus()), f"{label} LP not optimal"
        objs[label] = h.getInfo().objective_function_value
    assert abs(objs["rust"] - objs["python"]) < 1e-9
    assert abs(objs["rust"] - 700.0) < 1e-6


def test_lp_loaded_solution_aligned_by_names(tmp_path):
    """LP-loaded solutions must map values to the right variables even
    though zero-objective columns are reordered in the LP file."""
    from nimopt.solvers.highs import HiGHSSolver

    m = _dispatch_model()
    f = str(tmp_path / "m.lp")
    m.to_lp(f, use_rust=True)
    s = HiGHSSolver()
    s.read_lp(f)
    s.set_model(m)
    s.solve()
    sol = s.get_solution()
    pv = sol.variables["p"]
    gi = {g: i for i, g in enumerate(pv.elements[0])}
    # solar produces its availability (5,5,0); coal fills the rest (3,3,8)
    assert np.allclose(pv.values[gi["solar"]], [5.0, 5.0, 0.0])
    assert np.allclose(pv.values[gi["coal"]], [3.0, 3.0, 8.0])


def test_direct_solver_matches_reference():
    """Direct path solves the same model with identical objective."""
    m = _dispatch_model()
    result, sol = m.solve()
    assert abs(result.objective_value - 700.0) < 1e-6
    pv = sol.variables["p"]
    gi = {g: i for i, g in enumerate(pv.elements[0])}
    assert np.allclose(pv.values[gi["solar"]], [5.0, 5.0, 0.0])


def test_partial_dim_rhs_broadcast():
    """RHS covering a subset of a constraint's free dims (cap[G] for a
    constraint free over [G, T]) must be broadcast, not consumed as the
    first flat values. All four build paths must agree: direct rust,
    direct python, rust LP, python LP."""
    import tempfile

    import highspy

    from nimopt.solvers.highs_direct import HiGHSDirectSolver

    m = no.Model("rhsreg", sense="minimize")
    G = no.Set("G", ["g0", "g1", "g2"])
    T = no.Set("T", ["t0", "t1", "t2", "t3"])
    mc = no.Param("mc", [G], np.array([10.0, 20.0, 30.0]))
    cap = no.Param("cap", [G], np.array([4.0, 6.0, 8.0]))
    dem = no.Param("demand", [T], np.array([9.0, 12.0, 15.0, 18.0]))
    p = m.var("p", sets=[G, T], lb=0)
    m.eq("balance", no.Sum(G, p[G, T]) == dem[T])
    m.eq("max_output", p[G, T] <= cap[G])  # partial-dim RHS
    m.set_objective(no.Sum(G, T, mc[G] * p[G, T]))

    objs = {}
    for label, use_rust in (("direct_rust", True), ("direct_py", False)):
        s = HiGHSDirectSolver(use_rust=use_rust)
        s.load_model(m)
        r = s.solve()
        assert r.status.value == "optimal", f"{label}: {r.status}"
        objs[label] = r.objective_value
    for label, use_rust in (("lp_rust", True), ("lp_py", False)):
        f = tempfile.NamedTemporaryFile(suffix=".lp", delete=False).name
        m.to_lp(f, use_rust=use_rust)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(f)
        h.run()
        assert "Optimal" in str(h.getModelStatus()), f"{label} not optimal"
        objs[label] = h.getInfo().objective_function_value
    ref = objs["direct_py"]
    assert ref > 0
    for label, v in objs.items():
        assert abs(v - ref) < 1e-9, f"{label}: {v} != {ref}"
