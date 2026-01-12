"""HiGHS solver wrapper."""

from pathlib import Path

from .base import Solver, SolverResult, SolverStatus


class HiGHSSolver(Solver):
    """Wrapper for HiGHS solver SDK."""

    def __init__(self):
        try:
            import highspy
            self._h = highspy.Highs()
            self._h.setOptionValue("output_flag", False)
        except ImportError:
            raise ImportError("highspy not installed. Run: pip install highspy")

    def read_lp(self, path: str | Path) -> None:
        """Load model from LP file."""
        self._h.readModel(str(path))

    def read_mps(self, path: str | Path) -> None:
        """Load model from MPS file."""
        self._h.readModel(str(path))

    def set_option(self, name: str, value) -> None:
        """Set solver option."""
        self._h.setOptionValue(name, value)

    def solve(self) -> SolverResult:
        """Solve the model and return result."""
        self._h.run()

        status = self._map_status(self._h.getModelStatus())
        info = self._h.getInfo()

        return SolverResult(
            status=status,
            objective_value=info.objective_function_value,
            solve_time=self._h.getRunTime(),
            iterations=info.simplex_iteration_count,
            nodes=info.mip_node_count,
            gap=info.mip_gap if info.mip_node_count > 0 else None,
        )

    def _map_status(self, highs_status) -> SolverStatus:
        """Map HiGHS status to SolverStatus."""
        status_str = str(highs_status)
        if "Optimal" in status_str:
            return SolverStatus.OPTIMAL
        elif "Infeasible" in status_str:
            return SolverStatus.INFEASIBLE
        elif "Unbounded" in status_str:
            return SolverStatus.UNBOUNDED
        elif "Time" in status_str:
            return SolverStatus.TIME_LIMIT
        elif "Iteration" in status_str:
            return SolverStatus.ITERATION_LIMIT
        else:
            return SolverStatus.UNKNOWN

    def get_variable_names(self) -> list[str]:
        """Get all variable names in model order."""
        return self._h.allVariableNames()

    def get_constraint_names(self) -> list[str]:
        """Get all constraint names in model order."""
        n_rows = self._h.getNumRow()
        names = []
        for i in range(n_rows):
            status, name = self._h.getRowName(i)
            names.append(name)
        return names

    def get_variable_values(self) -> list[float]:
        """Get primal solution values for all variables."""
        return self._h.allVariableValues()

    def get_variable_duals(self) -> list[float]:
        """Get reduced costs for all variables."""
        return self._h.allVariableDuals()

    def get_constraint_duals(self) -> list[float]:
        """Get dual values (shadow prices) for all constraints."""
        return self._h.allConstrDuals()

    def write_solution(self, path: str | Path) -> None:
        """Write solution to file."""
        self._h.writeSolution(str(path), 0)
