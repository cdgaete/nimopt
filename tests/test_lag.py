"""Tests for lag/lead time index operations."""

import numpy as np

import nimopt as no
from nimopt.solvers import HiGHSDirectSolver


class TestLagBasic:
    """Basic lag/lead functionality."""

    def test_lagged_set_creation(self):
        """Test Set.lag() and Set.lead() create LaggedSet."""
        h = no.Set("h", [1, 2, 3, 4])

        h_lag = h.lag(1)
        assert h_lag.offset == -1
        assert h_lag.cyclic is False
        assert h_lag.base_set is h

        h_lead = h.lead(1)
        assert h_lead.offset == 1
        assert h_lead.cyclic is False

        h_lag_cyclic = h.lag(1, cyclic=True)
        assert h_lag_cyclic.cyclic is True

    def test_lagged_element_lookup(self):
        """Test get_lagged_element for various cases."""
        h = no.Set("h", [1, 2, 3, 4])

        # Lag by 1: h=2 -> h=1, h=1 -> None (out of bounds)
        h_lag = h.lag(1)
        assert h_lag.get_lagged_element(2) == 1
        assert h_lag.get_lagged_element(3) == 2
        assert h_lag.get_lagged_element(1) is None  # out of bounds

        # Lead by 1: h=2 -> h=3, h=4 -> None
        h_lead = h.lead(1)
        assert h_lead.get_lagged_element(2) == 3
        assert h_lead.get_lagged_element(4) is None

        # Cyclic lag: h=1 -> h=4
        h_lag_cyclic = h.lag(1, cyclic=True)
        assert h_lag_cyclic.get_lagged_element(1) == 4
        assert h_lag_cyclic.get_lagged_element(2) == 1

    def test_varref_with_lag(self):
        """Test VarRef recognizes lagged indices."""
        h = no.Set("h", [1, 2, 3])
        m = no.Model()
        x = m.var("x", [h])

        # Normal indexing
        ref = x[h]
        assert not ref.has_lag
        assert ref.lagged_indices == []

        # Lagged indexing
        ref_lag = x[h.lag(1)]
        assert ref_lag.has_lag
        assert len(ref_lag.lagged_indices) == 1
        assert ref_lag.lagged_indices[0][0] == 0  # position
        assert ref_lag.lagged_indices[0][1].offset == -1


class TestLagConstraints:
    """Test lagged constraints in optimization."""

    def test_simple_lag_constraint(self):
        """Test basic lag constraint: x[h] >= x[h-1]."""
        h = no.Set("h", [1, 2, 3, 4])

        m = no.Model(sense="minimize")
        x = m.var("x", [h], lb=0, ub=10)

        # x[h] >= x[h-1] for h > 1 (non-decreasing)
        # Rewrite as: x[h-1] <= x[h]
        m.eq("nondec", x[h.lag(1)] <= x[h])

        m.objective = no.Sum(h, x[h])

        solver = HiGHSDirectSolver(use_rust=False)
        solver.load_model(m)

        # Should have 3 nondec constraints (h=2,3,4 reference h=1,2,3)
        con_names = solver.get_constraint_names()
        nondec_cons = [c for c in con_names if c.startswith("nondec")]
        assert len(nondec_cons) == 3, f"Got {len(nondec_cons)}: {nondec_cons}"

    def test_cyclic_lag_constraint(self):
        """Test cyclic constraint: x[h] + x[h-1] >= 1 with wrap-around."""
        h = no.Set("h", [1, 2, 3])

        m = no.Model(sense="minimize")
        x = m.var("x", [h], lb=0, ub=1)

        # x[h] + x[h-1] >= 1, cyclic (h=1 references h=3)
        m.eq("pair", x[h] + x[h.lag(1, cyclic=True)] >= 1)

        m.objective = no.Sum(h, x[h])

        solver = HiGHSDirectSolver(use_rust=False)
        solver.load_model(m)
        result = solver.solve()

        assert result.status.name == "OPTIMAL"
        # All 3 constraints should be generated (cyclic has no boundary skip)
        assert len(solver.get_constraint_names()) == 3
        # Minimum is 1.5 (at least half of each pair must sum to 1)
        # Note: use computed objective from solution values (HiGHS binding issue)
        actual_obj = sum(solver.get_variable_values())
        assert actual_obj >= 1.5 - 1e-6

    def test_lead_constraint(self):
        """Test lead: x[h] <= x[h+1] (non-decreasing)."""
        h = no.Set("h", [1, 2, 3, 4])

        m = no.Model(sense="minimize")
        x = m.var("x", [h], lb=0, ub=100)

        # x[h] <= x[h+1]
        # For h=4, x[h+1] out of bounds -> constraint skipped
        m.eq("nondec", x[h] <= x[h.lead(1)])

        m.objective = no.Sum(h, x[h])

        solver = HiGHSDirectSolver(use_rust=False)
        solver.load_model(m)

        # Model should have 3 constraints (h=1,2,3), not 4
        assert len(solver.get_constraint_names()) == 3

    def test_multi_dim_lag(self):
        """Test lag on one dimension of multi-dimensional variable."""
        i = no.Set("i", ["a", "b"])
        h = no.Set("h", [1, 2, 3])

        m = no.Model()
        x = m.var("x", [i, h], lb=0)

        # x[i,h] >= x[i,h-1] for all i, h>1
        m.eq("nondec", x[i, h] >= x[i, h.lag(1)])

        solver = HiGHSDirectSolver(use_rust=False)
        solver.load_model(m)

        # Should have 2 (i) * 2 (h=2,3) = 4 constraints
        con_names = solver.get_constraint_names()
        assert len(con_names) == 4, f"Expected 4, got {len(con_names)}: {con_names}"


class TestLagWithRustFallback:
    """Test that lagged constraints correctly fall back to Python."""

    def test_rust_flag_no_effect_on_lag(self):
        """Lagged constraints should work regardless of use_rust flag."""
        h = no.Set("h", [1, 2, 3])
        m = no.Model()
        x = m.var("x", [h], lb=0, ub=10)

        # x[h] >= x[h-1] + 1
        m.eq("grow", x[h] >= x[h.lag(1)] + 1)

        m.objective = no.Sum(h, x[h])

        # Test with use_rust=True (should fall back to Python for lag)
        solver1 = HiGHSDirectSolver(use_rust=True)
        solver1.load_model(m)
        result1 = solver1.solve()

        # Test with use_rust=False
        solver2 = HiGHSDirectSolver(use_rust=False)
        solver2.load_model(m)
        result2 = solver2.solve()

        assert result1.status.name == "OPTIMAL"
        assert result2.status.name == "OPTIMAL"
        np.testing.assert_allclose(
            result1.objective_value, result2.objective_value, rtol=1e-6
        )
