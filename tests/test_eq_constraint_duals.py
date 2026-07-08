"""Regression: multi-term free EQUALITY constraints must expose per-row duals.

A multi-variable free constraint (e.g. `x[I] + y[I] == b`) is built by the
multi-term Rust fast path, which used to name its rows positionally
(`bal_0, bal_1, ...`) instead of by element (`bal_a, bal_b, ...`). Solution
extraction matches solver row names against element-based expected names, so
the positional names matched nothing -> the constraint got ZERO dual rows and,
worse, `con_offset` didn't advance, misaligning every later constraint's duals.
"""

import nimopt as no
from nimopt.solvers.highs_direct import HiGHSDirectSolver


def _solve(model):
    s = HiGHSDirectSolver()
    s.load_model(model)
    r = s.solve()
    return s, r, s.get_solution()


def test_multiterm_equality_constraint_has_full_duals():
    m = no.Model(name="eqdual", sense="minimize")
    I = no.Set("I", ["a", "b", "c"])
    x = m.var("x", sets=[I], lb=0, ub=8)
    y = m.var("y", sets=[I], lb=0, ub=8)

    # multi-term free equality -> multi-term Rust fast path
    m.eq("bal", x[I] + y[I] == 10)
    # a constraint AFTER the equality, to catch dual misalignment
    m.eq("xcap", x[I] <= 6)
    m.set_objective(no.Sum(I, x[I] + y[I]))

    _s, r, sol = _solve(m)
    assert str(r.status) == "SolverStatus.OPTIMAL"

    # The equality constraint must have one dual per element of I, not zero.
    bal = sol.constraints["bal"]
    assert bal.dims == ["I"], bal.dims
    assert bal.duals.size == 3, f"expected 3 dual rows, got {bal.duals.size}"

    # And the constraint after it must still have its full, correctly-sized duals
    # (proves con_offset stayed aligned).
    xcap = sol.constraints["xcap"]
    assert xcap.dims == ["I"]
    assert xcap.duals.size == 3
