import numpy as np
import pytest

import nimopt as no

pytest.importorskip("nimopt_rust")
highspy = pytest.importorskip("highspy")


def _pinned_objective_model():
    WR = no.Set("WR", ["r1", "r2"])
    T = no.Set("T", [1, 2, 3])
    m = no.Model("obj_pin", sense="maximize")
    vol = m.var("vol", [WR, T], lb=0, ub=5)
    m.eq("cap", no.Sum(WR, T, vol[WR, T]) <= 20)
    m.set_objective(no.Sum(WR, vol[WR, 1]))
    return m


def test_solve_honors_a_literal_index_in_the_objective():
    res, _ = _pinned_objective_model().solve()
    assert res.objective_value == pytest.approx(10.0)


@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_objective_keeps_a_literal_index_pinned(tmp_path, use_rust):
    path = tmp_path / f"obj_{use_rust}.lp"
    _pinned_objective_model().to_lp(str(path), use_rust=use_rust)

    text = path.read_text()
    row = next(r for r in text.splitlines() if r.strip().startswith("obj:"))
    assert [t for t in row.split() if t.startswith("vol_")] == ["vol_r1_1", "vol_r2_1"]


def test_lp_objective_and_solve_agree_on_a_literal_index(tmp_path):
    m = _pinned_objective_model()
    path = tmp_path / "obj.lp"
    m.to_lp(str(path))
    res, _ = m.solve()

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    h.run()

    assert h.getObjectiveValue() == pytest.approx(res.objective_value)


def _array_coef_model():
    G = no.Set("G", ["a", "b"])
    T = no.Set("T", [1, 2, 3])
    w = no.Param("w", [G, T], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    m = no.Model("obj_coef", sense="maximize")
    p = m.var("p", [G, T], lb=0, ub=5)
    m.eq("cap", no.Sum(G, T, p[G, T]) <= 20)
    m.set_objective(no.Sum(G, w[G, 2] * p[G, 2]))
    return m


@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_objective_indexes_an_array_coef_at_the_literal_element(tmp_path, use_rust):
    path = tmp_path / f"obj_coef_{use_rust}.lp"
    _array_coef_model().to_lp(str(path), use_rust=use_rust)

    row = next(r for r in path.read_text().splitlines() if r.strip().startswith("obj:"))
    terms = row.split(":")[1].split()
    assert [t for t in terms if t.startswith("p_")] == ["p_a_2", "p_b_2"]
    assert [float(t) for t in terms if not t.startswith(("p_", "+"))] == [2.0, 5.0]


def test_solve_indexes_an_array_coef_at_the_literal_element():
    res, _ = _array_coef_model().solve()
    assert res.objective_value == pytest.approx(35.0)


def test_an_unpinned_objective_term_is_unaffected():
    G = no.Set("G", ["a", "b"])
    m = no.Model("plain", sense="maximize")
    x = m.var("x", [G], lb=0, ub=3)
    m.eq("cap", no.Sum(G, x[G]) <= 4)
    m.set_objective(no.Sum(G, x[G]))
    res, _ = m.solve()
    assert res.objective_value == pytest.approx(4.0)


@pytest.mark.parametrize("builder", ["rust", "python"])
def test_objective_vector_pins_a_literal_index_in_both_builders(builder):
    from nimopt.solvers.highs_direct import (
        _build_matrices_python,
        _build_matrices_rust,
        _generate_var_names_from_info,
    )

    m = _pinned_objective_model()
    if builder == "rust":
        mat = _build_matrices_rust(m)
        names = _generate_var_names_from_info(mat["var_info"])
    else:
        mat = _build_matrices_python(m)
        names = mat["var_names"]

    obj = {n: float(v) for n, v in zip(names, mat["c"]) if v}
    assert obj == {"vol_r1_1": 1.0, "vol_r2_1": 1.0}
