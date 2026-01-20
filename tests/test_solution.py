"""Tests for solution extraction."""

import tempfile
from pathlib import Path

import pytest

import nimopt as no
from nimopt.solution import extract_solution_python, write_solution_csv
from nimopt.solvers import HiGHSSolver


@pytest.fixture
def transport_model():
    """Create transport model and return (model, solver)."""
    i = no.Set("i", ["seattle", "sandiego"])
    j = no.Set("j", ["newyork", "chicago", "topeka"])
    supply = no.Param("supply", [i], [350, 600])
    demand = no.Param("demand", [j], [325, 300, 275])
    cost = no.Param("cost", [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

    m = no.Model(name="transport", sense="minimize")
    x = m.var("x", [i, j], lb=0)
    m.eq("supply", no.Sum(j, x[i, j]) <= supply[i])
    m.eq("demand", no.Sum(i, x[i, j]) >= demand[j])
    m.objective = no.Sum(i, j, cost[i, j] * x[i, j])

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        lp_file = f.name
    m.to_lp(lp_file)

    solver = HiGHSSolver()
    solver.read_lp(lp_file)
    solver.solve()

    return m, solver


def test_extract_solution_shapes(transport_model):
    """Test solution arrays have correct shapes."""
    m, solver = transport_model
    sol = extract_solution_python(solver, m)

    # Variable x has shape (2, 3)
    assert "x" in sol.variables
    assert sol.variables["x"].values.shape == (2, 3)
    assert sol.variables["x"].duals.shape == (2, 3)
    assert sol.variables["x"].dims == ["i", "j"]

    # Constraint supply has shape (2,)
    assert "supply" in sol.constraints
    assert sol.constraints["supply"].duals.shape == (2,)
    assert sol.constraints["supply"].dims == ["i"]

    # Constraint demand has shape (3,)
    assert "demand" in sol.constraints
    assert sol.constraints["demand"].duals.shape == (3,)
    assert sol.constraints["demand"].dims == ["j"]


def test_extract_solution_values(transport_model):
    """Test solution values are correct."""
    m, solver = transport_model
    sol = extract_solution_python(solver, m)

    # Check optimal shipments (known solution)
    x = sol.variables["x"].values
    assert x[0, 1] == pytest.approx(300)  # seattle -> chicago
    assert x[1, 2] == pytest.approx(275)  # sandiego -> topeka


def test_write_solution_csv_rust(transport_model):
    """Test Rust CSV writer produces correct output."""
    m, solver = transport_model
    sol = extract_solution_python(solver, m)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_solution_csv(sol, tmpdir, use_rust=True)

        # Check variable CSV
        var_path = Path(tmpdir) / "var_x.csv"
        assert var_path.exists()
        content = var_path.read_text()
        assert "i,j,value,dual" in content
        assert "seattle,chicago,300" in content

        # Check constraint CSV
        con_path = Path(tmpdir) / "con_demand.csv"
        assert con_path.exists()
        content = con_path.read_text()
        assert "j,dual" in content


def test_write_solution_csv_python(transport_model):
    """Test Python CSV writer produces correct output."""
    m, solver = transport_model
    sol = extract_solution_python(solver, m)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_solution_csv(sol, tmpdir, use_rust=False)

        var_path = Path(tmpdir) / "var_x.csv"
        assert var_path.exists()
        content = var_path.read_text()
        assert "i,j,value,dual" in content
