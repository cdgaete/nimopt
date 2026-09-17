import warnings

import numpy as np
import pytest

import nimopt as no

linopy = pytest.importorskip("linopy", reason="the comparison needs the bench extra")
xr = pytest.importorskip("xarray")

GENERATORS = ["g0", "g1", "g2"]
PERIODS = ["t0", "t1", "t2"]
BREAKPOINTS = ["b0", "b1", "b2", "b3"]
DEMAND = np.array([40.0, 90.0, 130.0])
FLAT = np.array([0.0, 0.0, 5.0])
FIXED = np.array([3.0, 4.0, 0.0])
NAN = np.nan

FULL_X = np.array([[0, 30, 60, 100], [0, 20, 50, 80], [10, 20, 30, 40.0]])
NOT_CONVEX = np.array([[0, 60, 75, 160], [0, 50, 80, 170], [0, 5, 20, 60.0]])
CONVEX = np.array([[0, 30, 90, 200], [0, 10, 40, 100], [0, 5, 20, 60.0]])
CONCAVE = np.array([[0, 90, 150, 170], [0, 40, 60, 70], [0, 50, 90, 100.0]])
# g1 has three breakpoints and g2 none
SHORT_X = np.array([[0, 30, 60, 100], [0, 20, 50, NAN], [NAN] * 4])
SHORT_Y = np.array([[0, 60, 75, 160], [0, 50, 80, NAN], [NAN] * 4])


def nimopt_side(xp, yp, sign, method, active):
    G = no.Set("G", np.array(GENERATORS))
    T = no.Set("T", np.array(PERIODS))
    B = no.Set("B", np.array(BREAKPOINTS))
    at = np.nonzero(~np.isnan(xp))
    labels = {"G": np.array(GENERATORS)[at[0]], "B": np.array(BREAKPOINTS)[at[1]]}
    X = no.Param.from_long("xp", (G, B), labels, xp[at])
    Y = no.Param.from_long("yp", (G, B), labels, yp[at])
    m = no.Model("pwl")
    p = m.var("p", (G, T), upper=200.0)
    c = m.var("c", (G, T), lower=-1e4)
    flat = no.Param.from_dense("flat", (G,), FLAT)
    objective = no.Sum(G, T, c[G, T]) + no.Sum(G, T, flat[G] * p[G, T])
    on = None
    if active:
        u = m.var("u", (G, T), upper=1.0, integer=True)
        fixed = no.Param.from_dense("fixed", (G,), FIXED)
        objective = objective + no.Sum(G, T, fixed[G] * u[G, T])
        on = u[G, T]
    m.piecewise("cost", p[G, T], X[G, B], c[G, T], Y[G, B], sign, method, on)
    demand = no.Param.from_dense("demand", (T,), DEMAND)
    m.constraint("demand", no.Sum(G, p[G, T]) == demand[T])
    m.set_objective(objective)
    assembled = m.assemble()
    solved = m.solve()
    return {
        "objective": solved.objective,
        "rows": assembled.n_rows,
        "cols": assembled.n_cols,
        "nonzeros": int(np.count_nonzero(assembled.values)),
    }


def linopy_side(xp, yp, sign, method, active):
    m = linopy.Model()
    coords = [GENERATORS, PERIODS]
    p = m.add_variables(lower=0, upper=200, coords=coords, dims=["G", "T"], name="p")
    c = m.add_variables(lower=-1e4, coords=coords, dims=["G", "T"], name="c")
    has = ~np.isnan(xp).all(axis=1)
    names = [g for g, h in zip(GENERATORS, has) if h]
    points = [names, range(len(BREAKPOINTS))]
    bx = xr.DataArray(xp[has], coords=points, dims=["G", "_breakpoint"])
    by = xr.DataArray(yp[has], coords=points, dims=["G", "_breakpoint"])
    flat = xr.DataArray(FLAT, coords=[GENERATORS], dims=["G"])
    objective = c.sum() + (flat * p).sum()
    extra = {}
    if active:
        u = m.add_variables(binary=True, coords=coords, dims=["G", "T"], name="u")
        fixed = xr.DataArray(FIXED, coords=[GENERATORS], dims=["G"])
        objective = objective + (fixed * u).sum()
        extra["active"] = u.sel(G=names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linopy.EvolvingAPIWarning)
        m.add_piecewise_formulation(
            (c.sel(G=names), by, sign),
            (p.sel(G=names), bx),
            method={"tangent": "lp"}.get(method, method),
            **extra,
        )
    demand = xr.DataArray(DEMAND, coords=[PERIODS], dims=["T"])
    m.add_constraints(p.sum("G") == demand)
    m.add_objective(objective)
    m.solve("highs", output_flag=False)
    A = m.matrices.A
    return {
        "objective": m.objective.value,
        "rows": A.shape[0],
        "cols": A.shape[1],
        "nonzeros": A.nnz,
    }


FULL = [
    ("incremental", ">=", NOT_CONVEX),
    ("incremental", "==", NOT_CONVEX),
    ("incremental", "<=", NOT_CONVEX),
    ("tangent", ">=", CONVEX),
    ("tangent", "<=", CONCAVE),
]


@pytest.mark.parametrize(("method", "sign", "yp"), FULL)
def test_every_breakpoint_present_builds_the_problem_linopy_builds(method, sign, yp):
    ours = nimopt_side(FULL_X, yp, sign, method, active=False)
    theirs = linopy_side(FULL_X, yp, sign, method, active=False)
    assert ours["objective"] == pytest.approx(theirs["objective"], rel=1e-9)
    for field in ("rows", "cols", "nonzeros"):
        assert ours[field] == theirs[field], field


@pytest.mark.parametrize("sign", [">=", "=="])
def test_active_builds_the_columns_and_nonzeros_linopy_builds(sign):
    # nimopt stores an explicit zero where a first breakpoint is 0; the
    # count of nonzero values matches
    ours = nimopt_side(FULL_X, NOT_CONVEX, sign, "incremental", active=True)
    theirs = linopy_side(FULL_X, NOT_CONVEX, sign, "incremental", active=True)
    assert ours["objective"] == pytest.approx(theirs["objective"], rel=1e-9)
    assert ours["cols"] == theirs["cols"]
    assert ours["nonzeros"] == theirs["nonzeros"]


@pytest.mark.parametrize("active", [False, True])
def test_absent_breakpoints_solve_to_the_objective_linopy_reports(active):
    # linopy keeps a one-term row at an absent segment, and nimopt does not
    ours = nimopt_side(SHORT_X, SHORT_Y, ">=", "incremental", active)
    theirs = linopy_side(SHORT_X, SHORT_Y, ">=", "incremental", active)
    assert ours["objective"] == pytest.approx(theirs["objective"], rel=1e-9)
    assert ours["cols"] == theirs["cols"]
    assert ours["rows"] < theirs["rows"]


def test_tangent_with_absent_breakpoints_builds_the_problem_linopy_builds():
    convex = np.array([[0, 30, 90, 200], [0, 30, 90, NAN], [NAN] * 4])
    ours = nimopt_side(SHORT_X, convex, ">=", "tangent", active=False)
    theirs = linopy_side(SHORT_X, convex, ">=", "tangent", active=False)
    assert ours["objective"] == pytest.approx(theirs["objective"], rel=1e-9)
    for field in ("rows", "cols", "nonzeros"):
        assert ours[field] == theirs[field], field
