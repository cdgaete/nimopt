"""Test coefficient dimension mismatch - LP Rust writer bug fix."""


def test_coef_more_dims_than_var_rust(tmp_path):
    """Test coefficient with more dimensions than variable.
    
    Regression test for LP Rust writer bug where Avail[Tech, Hours] * N[Tech]
    produced incorrect coefficients because the Rust code didn't correctly
    index into coefficients when they had more dimensions than the variable.
    """
    import nimopt as no
    from nimopt import Sum

    Hours = no.Set("Hours", [1, 2, 3])
    Tech = no.Set("Tech", ["A", "B"])
    Avail = no.Param(
        "Avail", [Tech, Hours], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    m = no.Model()
    G = m.var("G", [Tech, Hours], lb=0)
    N = m.var("N", [Tech], lb=0)
    m.objective = Sum(Tech, Hours, G[Tech, Hours])
    m.eq("MaxGen", G[Tech, Hours] == Avail[Tech, Hours] * N[Tech])

    # Write with Python
    py_lp = tmp_path / "py.lp"
    m.to_lp(str(py_lp), use_rust=False)

    # Write with Rust
    rust_lp = tmp_path / "rust.lp"
    m.to_lp(str(rust_lp), use_rust=True)

    # Extract MaxGen constraints
    py_lines = [l for l in py_lp.read_text().splitlines() if "MaxGen" in l]
    rust_lines = [l for l in rust_lp.read_text().splitlines() if "MaxGen" in l]

    # Both should have 6 constraints
    assert len(py_lines) == 6
    assert len(rust_lines) == 6

    # Extract coefficients for N from each constraint
    # Lines look like: " MaxGen_0: G_A_1 - 0.1 N_A = -0.0"
    import re

    def extract_n_coef(line):
        # Match coefficient before N_
        match = re.search(r"[-+]?\s*([\d.]+)\s*N_", line)
        if match:
            return float(match.group(1))
        # Check for "- N_" (coefficient of -1)
        if "- N_" in line:
            return 1.0
        return None

    py_coefs = [extract_n_coef(l) for l in py_lines]
    rust_coefs = [extract_n_coef(l) for l in rust_lines]

    # Should be [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for i, (exp, py_c, rust_c) in enumerate(zip(expected, py_coefs, rust_coefs)):
        assert abs(py_c - exp) < 1e-9, f"Python constraint {i}: expected {exp}, got {py_c}"
        assert abs(rust_c - exp) < 1e-9, f"Rust constraint {i}: expected {exp}, got {rust_c}"


def test_coef_more_dims_solve():
    """Test that the model solves correctly with the fix."""
    import nimopt as no
    from nimopt import Sum
    from nimopt.solvers import HiGHSDirectSolver

    Hours = no.Set("Hours", [1, 2])
    Tech = no.Set("Tech", ["A", "B"])
    # Availability: Tech A has 50%, 60%; Tech B has 70%, 80%
    Avail = no.Param("Avail", [Tech, Hours], [[0.5, 0.6], [0.7, 0.8]])
    
    # Demand per hour
    Demand = no.Param("Demand", [Hours], [100, 120])

    m = no.Model(sense="minimize")
    G = m.var("G", [Tech, Hours], lb=0)  # Generation
    N = m.var("N", [Tech], lb=0)  # Installed capacity
    
    # Minimize total capacity
    m.objective = Sum(Tech, N[Tech])
    
    # Generation must meet demand
    m.eq("MeetDemand", Sum(Tech, G[Tech, Hours]) >= Demand[Hours])
    
    # Generation cannot exceed available capacity
    m.eq("MaxGen", G[Tech, Hours] <= Avail[Tech, Hours] * N[Tech])

    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(m)
    result = solver.solve()

    assert result.status.value == "optimal"
    
    sol = solver.get_solution()
    N_vals = sol.var("N")
    
    # With the bug, the model would be infeasible or give wrong answers
    # because MaxGen constraints would use wrong coefficients
    
    # Check that installed capacity is positive and reasonable
    total_cap = float(N_vals.values.sum())
    assert total_cap > 0
    assert total_cap < 1000  # Sanity check - shouldn't need huge capacity
