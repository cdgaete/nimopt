import re
import time

import numpy as np
import pytest

import nimopt as no

pytest.importorskip("nimopt_rust")


_TERM_RE = re.compile(r"([+-]?)\s*([0-9.eE+-]*)\s*(p_[A-Za-z0-9_]+)")


def _row_coefs(row):
    """Coefficients of an LP row in order; a bare variable means 1."""
    out = []
    for sign, num, _var in _TERM_RE.findall(row.split(":")[1]):
        val = float(num) if num else 1.0
        out.append(-val if sign == "-" else val)
    return tuple(out)


def _model(n_rows):
    G = no.Set("G", [f"g{i}" for i in range(n_rows)])
    T = no.Set("T", list(range(20)))
    w = no.Param("w", [G, T], np.arange(n_rows * 20, dtype=float).reshape(n_rows, 20))
    m = no.Model("scale", sense="minimize")
    p = m.var("p", [G, T], lb=0, ub=10)
    # Pinned index keeps this on the batch slow path, one array-coef term per row.
    m.eq("c", w[G, 0] * p[G, 0] >= 1.0)
    m.set_objective(no.Sum(G, no.Sum(T, p[G, T])))
    return m


def _export_seconds(n_rows, tmp_path):
    m = _model(n_rows)
    path = str(tmp_path / f"s{n_rows}.lp")
    m.to_lp(path)  # warm any per-model caching
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        m.to_lp(path)
        best = min(best, time.perf_counter() - t0)
    return best


def test_slow_path_export_scales_linearly_in_row_count(tmp_path):
    small = _export_seconds(500, tmp_path)
    large = _export_seconds(4000, tmp_path)

    # 8x the rows. Linear would be ~8x; the quadratic coordinate lookup made it
    # ~60x. Bound generously so this measures the class, not the machine.
    assert large / small < 16, f"{small=:.4f}s {large=:.4f}s ratio={large / small:.1f}"


@pytest.mark.parametrize("use_rust", [True, False])
def test_array_coef_is_read_at_the_right_cell(tmp_path, use_rust):
    G = no.Set("G", ["a", "b"])
    T = no.Set("T", [1, 2, 3])
    w = no.Param("w", [G, T], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    m = no.Model("cells", sense="minimize")
    p = m.var("p", [G, T], lb=0, ub=10)
    m.eq("c", no.Sum(G, w[G, T] * p[G, T]) >= 1.0)
    m.set_objective(no.Sum(G, T, w[G, T] * p[G, T]))

    path = tmp_path / f"cells_{use_rust}.lp"
    m.to_lp(str(path), use_rust=use_rust)

    rows = [r for r in path.read_text().splitlines() if r.strip().startswith("c_")]
    assert len(rows) == 3
    for j, expected in enumerate([(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]):
        assert _row_coefs(rows[j]) == expected
