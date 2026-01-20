"""Tests for HiGHS direct solver (no LP file)."""


def test_direct_solver_transport():
    """Test direct solver with transport problem."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver, SolverStatus

    # Create transport model
    i = no.Set("i", ["seattle", "sandiego"])
    j = no.Set("j", ["newyork", "chicago", "topeka"])

    a = no.Param("a", [i], [350, 600])
    b = no.Param("b", [j], [325, 300, 275])
    d = no.Param("d", [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

    m = no.Model(name="transport", sense="minimize")
    x = m.var("x", [i, j], lb=0)

    m.set_objective(no.Sum(i, j, d[i, j] * x[i, j]))
    m.eq("supply", no.Sum(j, x[i, j]) <= a[i])
    m.eq("demand", no.Sum(i, x[i, j]) >= b[j])

    # Solve directly
    solver = HiGHSDirectSolver()
    solver.load_model(m)
    result = solver.solve()

    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 1707.5) < 1e-4

    # Check solution values
    values = solver.get_variable_values()
    assert len(values) == 6  # 2 sources x 3 destinations


def test_direct_solver_simple():
    """Test direct solver with a simple 2-variable problem."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver, SolverStatus

    # min x + 2y
    # s.t. x + y >= 10
    #      x >= 0, y >= 0
    m = no.Model(sense="minimize")
    x = m.var("x", lb=0)
    y = m.var("y", lb=0)

    m.set_objective(x + 2 * y)
    m.eq("con1", x + y >= 10)

    solver = HiGHSDirectSolver()
    solver.load_model(m)
    result = solver.solve()

    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 10.0) < 1e-6  # x=10, y=0

    values = solver.get_variable_values()
    assert abs(values[0] - 10.0) < 1e-6  # x = 10
    assert abs(values[1] - 0.0) < 1e-6  # y = 0


def test_direct_vs_lp_file():
    """Verify direct solver gives same result as LP file solver."""
    import os
    import tempfile

    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver, HiGHSSolver, SolverStatus

    # Create model
    i = no.Set("i", ["a", "b", "c"])
    j = no.Set("j", [1, 2, 3, 4])

    cost = no.Param(
        "cost",
        [i, j],
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
    )
    cap = no.Param("cap", [i], [100, 200, 150])
    dem = no.Param("dem", [j], [80, 90, 70, 110])

    m = no.Model(sense="minimize")
    x = m.var("x", [i, j], lb=0)

    m.set_objective(no.Sum(i, j, cost[i, j] * x[i, j]))
    m.eq("capacity", no.Sum(j, x[i, j]) <= cap[i])
    m.eq("demand", no.Sum(i, x[i, j]) >= dem[j])

    # Solve with LP file
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        lp_path = f.name

    try:
        m.to_lp(lp_path)
        solver_lp = HiGHSSolver()
        solver_lp.read_lp(lp_path)
        result_lp = solver_lp.solve()
    finally:
        os.unlink(lp_path)

    # Solve directly
    solver_direct = HiGHSDirectSolver()
    solver_direct.load_model(m)
    result_direct = solver_direct.solve()

    # Compare results
    assert result_lp.status == result_direct.status == SolverStatus.OPTIMAL
    assert abs(result_lp.objective_value - result_direct.objective_value) < 1e-6


def test_direct_solver_maximize():
    """Test direct solver with maximization."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver, SolverStatus

    # max 3x + 2y
    # s.t. x + y <= 10
    #      x <= 6
    #      y <= 5
    #      x >= 0, y >= 0
    m = no.Model(sense="maximize")
    x = m.var("x", lb=0, ub=6)
    y = m.var("y", lb=0, ub=5)

    m.set_objective(3 * x + 2 * y)
    m.eq("total", x + y <= 10)

    solver = HiGHSDirectSolver()
    solver.load_model(m)
    result = solver.solve()

    assert result.status == SolverStatus.OPTIMAL
    # Optimal: x=6, y=4 -> 3*6 + 2*4 = 26
    assert abs(result.objective_value - 26.0) < 1e-6


def test_direct_solver_equality():
    """Test direct solver with equality constraints."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver, SolverStatus

    # min x + y
    # s.t. x + y = 10
    #      x >= 0, y >= 0
    m = no.Model(sense="minimize")
    x = m.var("x", lb=0)
    y = m.var("y", lb=0)

    m.set_objective(x + y)
    m.eq("equal", x + y == 10)

    solver = HiGHSDirectSolver()
    solver.load_model(m)
    result = solver.solve()

    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 10.0) < 1e-6


def test_direct_solver_solution_arrays():
    """Test solution extraction as nimblend Arrays."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver

    i = no.Set('i', ['a', 'b'])
    j = no.Set('j', ['x', 'y', 'z'])

    m = no.Model(sense='minimize')
    v = m.var('v', [i, j], lb=0, ub=10)
    m.set_objective(no.Sum(i, j, v[i, j]))
    m.eq('limit', no.Sum(j, v[i, j]) >= 5)

    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(m)
    solver.solve()

    # Get solution
    sol = solver.get_solution()

    # Check objective
    assert sol.objective_value is not None
    assert abs(sol.objective_value - 10.0) < 1e-6

    # Check variable array
    v_arr = sol.var('v')
    assert v_arr.shape == (2, 3)
    assert v_arr.dims == ['i', 'j']
    assert list(v_arr.coords['i']) == ['a', 'b']
    assert list(v_arr.coords['j']) == ['x', 'y', 'z']

    # Check constraint duals
    limit_arr = sol.con('limit')
    assert limit_arr.shape == (2,)
    assert limit_arr.dims == ['i']
