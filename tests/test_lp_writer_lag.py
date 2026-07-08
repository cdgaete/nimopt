"""LP-writer regressions for lagged constraints and expression-RHS constants.

Two defects, both in how a constraint is *serialized* to LP (the direct solver
builds the same models correctly):

1. The rust writer's lagged slow path never applied the lag offset (a
   `# TODO: handle lag`), so `s[T] == s[T-1] + 1` was written as `s_T - s_T`
   for every T (including out-of-bounds T=1) instead of `s_T - s_{T-1}` for
   T = 2..N.

2. Both writers negated a numeric constant sitting on a LinearExpr RHS
   (`rhs_const = -con.rhs.const`), so `x == y + 5` was written as `x - y = -5`
   instead of `x - y = 5`. The direct solver keeps it un-negated.

Ground truth (direct solver): `s[T] == s[T-1] + 1` with s[1] pinned solves to
s = [1, 2, 3, 4], i.e. each row is `s_T - s_{T-1} = 1`.
"""

import tempfile
from pathlib import Path

import nimopt as no


def _write(model, use_rust):
    f = tempfile.mktemp(suffix=".lp")
    model.to_lp(f, use_rust=use_rust)
    text = Path(f).read_text()
    Path(f).unlink()
    return text


def _subject_rows(lp_text):
    """Return normalized constraint rows (between 'Subject To' and 'Bounds')."""
    rows = []
    in_st = False
    for line in lp_text.splitlines():
        s = line.strip()
        if s == "Subject To":
            in_st = True
            continue
        if s == "Bounds":
            break
        if in_st and s:
            rows.append(" ".join(s.split()))
    return rows


def _lag_model():
    T = no.Set("T", [1, 2, 3, 4])
    m = no.Model(sense="minimize")
    s = m.var("s", [T], lb=0, ub=100)
    m.eq("growth", s[T] == s[T.lag(1)] + 1)
    m.set_objective(no.Sum(T, s[T]))
    return m


def test_rust_writer_lag_offset_and_rows():
    rows = _subject_rows(_write(_lag_model(), use_rust=True))
    growth = [r for r in rows if r.split(":")[0].startswith("growth")]

    # exactly 3 rows (T=2,3,4); the out-of-bounds T=1 row must be skipped
    assert len(growth) == 3, growth
    # lag offset applied: consecutive-index differences, never `s_X - s_X`
    bodies = [r.split(":", 1)[1].strip() for r in growth]
    assert any(b.startswith("s_2 - s_1") for b in bodies), bodies
    assert any(b.startswith("s_3 - s_2") for b in bodies), bodies
    assert any(b.startswith("s_4 - s_3") for b in bodies), bodies
    assert not any("s_1 - s_1" in b for b in bodies), bodies


def test_rust_writer_lag_rhs_sign():
    rows = _subject_rows(_write(_lag_model(), use_rust=True))
    growth = [r for r in rows if r.split(":")[0].startswith("growth")]
    for r in growth:
        rhs = r.split("=")[-1].strip()
        assert float(rhs) == 1.0, r  # s_T - s_{T-1} = 1, not -1


def test_rust_writer_expr_rhs_const_sign():
    """x == y + 5  =>  x - y = 5 (not -5)."""
    m = no.Model(sense="minimize")
    x = m.var("x", [], lb=0, ub=100)
    y = m.var("y", [], lb=0, ub=100)
    m.eq("c", x == y + 5)
    m.set_objective(x + y)

    rows = _subject_rows(_write(m, use_rust=True))
    crow = next(r for r in rows if r.startswith("c:"))
    assert float(crow.split("=")[-1].strip()) == 5.0, crow


def test_py_writer_lag_rhs_sign():
    """The python writer already applies the lag offset; only its sign was off."""
    rows = _subject_rows(_write(_lag_model(), use_rust=False))
    growth = [r for r in rows if r.split(":")[0].startswith("growth")]
    assert len(growth) == 3, growth
    for r in growth:
        rhs = r.split("=")[-1].strip()
        assert float(rhs) == 1.0, r


def _lag_array_rhs_model():
    """`soc[H] == soc[H-1] + xfix[H]`: a lag term AND an ARRAY const on the RHS.

    The RHS is a LinearExpr whose `.const` is the param Array `xfix`, so the
    writer must index that array per free-index binding, not stash the whole
    array as one row's RHS.
    """
    H = no.Set("H", ["h1", "h2", "h3", "h4"])
    xfix = no.Param("xfix", [H], [2, 3, 4, 5])
    m = no.Model(sense="minimize")
    soc = m.var("soc", [H], lb=-1000, ub=1000)
    m.eq("socdef", soc[H] == soc[H.lag(1)] + xfix[H])
    m.set_objective(no.Sum(H, soc[H]))
    return m


def test_rust_writer_lag_array_rhs_const():
    """Array const on a lagged RHS: rows T=2,3,4 carry xfix[h2],xfix[h3],xfix[h4]."""
    rows = _subject_rows(_write(_lag_array_rhs_model(), use_rust=True))
    socdef = [r for r in rows if r.split(":")[0].startswith("socdef")]
    assert len(socdef) == 3, socdef  # h1 row skipped (lag out of bounds)
    rhs_vals = sorted(float(r.split("=")[-1].strip()) for r in socdef)
    assert rhs_vals == [3.0, 4.0, 5.0], socdef


def test_py_writer_lag_array_rhs_const():
    rows = _subject_rows(_write(_lag_array_rhs_model(), use_rust=False))
    socdef = [r for r in rows if r.split(":")[0].startswith("socdef")]
    assert len(socdef) == 3, socdef
    rhs_vals = sorted(float(r.split("=")[-1].strip()) for r in socdef)
    assert rhs_vals == [3.0, 4.0, 5.0], socdef
