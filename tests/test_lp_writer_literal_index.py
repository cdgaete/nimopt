import numpy as np
import pytest

import nimopt as no

pytest.importorskip("nimopt_rust")
highspy = pytest.importorskip("highspy")


def _pinned_model():
    WR = no.Set("WR", ["r1", "r2"])
    T = no.Set("T", [1, 2, 3])
    m = no.Model("pin", sense="maximize")
    vol = m.var("vol", [WR, T], lb=0, ub=100)
    m.eq("c", vol[WR, 1] <= 5)
    m.set_objective(no.Sum(WR, T, vol[WR, T]))
    return m


def _constraint_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("c_"):
            rows.append(line)
    return rows


@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_export_keeps_a_literal_index_pinned(tmp_path, use_rust):
    path = tmp_path / f"pin_{use_rust}.lp"
    _pinned_model().to_lp(str(path), use_rust=use_rust)

    rows = _constraint_rows(path)
    assert len(rows) == 2
    assert "vol_r1_2" not in " ".join(rows)
    assert "vol_r1_3" not in " ".join(rows)


def test_lp_export_and_solve_agree_on_a_literal_indexed_row(tmp_path):
    m = _pinned_model()
    path = tmp_path / "pin.lp"
    m.to_lp(str(path))
    res, _ = m.solve()

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    h.run()

    assert h.getObjectiveValue() == pytest.approx(res.objective_value)


@pytest.mark.parametrize("use_rust", [True, False])
def test_lp_export_indexes_an_array_coef_at_the_literal_element(tmp_path, use_rust):
    G = no.Set("G", ["a", "b"])
    T = no.Set("T", [1, 2, 3])
    w = no.Param("w", [G, T], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    m = no.Model("coef", sense="maximize")
    p = m.var("p", [G, T], lb=0, ub=10)
    m.eq("cap", no.Sum(G, w[G, T] * p[G, T]) <= 30)
    m.eq("pin", no.Sum(G, w[G, 2] * p[G, 2]) <= 12)
    m.set_objective(no.Sum(G, T, p[G, T]))

    path = tmp_path / f"coef_{use_rust}.lp"
    m.to_lp(str(path), use_rust=use_rust)

    row = next(r for r in path.read_text().splitlines() if r.strip().startswith("pin"))
    terms = row.split(":")[1].split("<=")[0].split()
    assert [t for t in terms if t.startswith("p_")] == ["p_a_2", "p_b_2"]
    assert [float(t) for t in terms if not t.startswith(("p_", "+"))] == [2.0, 5.0]

    res, _ = m.solve()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    h.run()
    assert h.getObjectiveValue() == pytest.approx(res.objective_value)
