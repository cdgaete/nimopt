"""Abstract base class for solver wrappers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class SolverStatus(Enum):
    """Solver termination status."""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SolverResult:
    """Result from solver execution."""
    status: SolverStatus
    objective_value: Optional[float] = None
    solve_time: float = 0.0
    iterations: int = 0
    nodes: int = 0  # For MIP
    gap: Optional[float] = None  # For MIP


class Solver(ABC):
    """Abstract base class for solver wrappers.

    Each solver implementation wraps a specific solver SDK (HiGHS, Gurobi, etc.)
    and provides a consistent interface for:
    - Loading models from LP/MPS files
    - Setting solver options
    - Solving and retrieving results
    - Extracting solution values and sensitivity data
    """

    @abstractmethod
    def read_lp(self, path: str | Path) -> None:
        """Load model from LP file."""
        pass

    @abstractmethod
    def read_mps(self, path: str | Path) -> None:
        """Load model from MPS file."""
        pass

    @abstractmethod
    def set_option(self, name: str, value) -> None:
        """Set solver option."""
        pass

    @abstractmethod
    def solve(self) -> SolverResult:
        """Solve the model and return result."""
        pass

    @abstractmethod
    def get_variable_names(self) -> list[str]:
        """Get all variable names in model order."""
        pass

    @abstractmethod
    def get_constraint_names(self) -> list[str]:
        """Get all constraint names in model order."""
        pass

    @abstractmethod
    def get_variable_values(self) -> list[float]:
        """Get primal solution values for all variables."""
        pass

    @abstractmethod
    def get_variable_duals(self) -> list[float]:
        """Get reduced costs for all variables."""
        pass

    @abstractmethod
    def get_constraint_duals(self) -> list[float]:
        """Get dual values (shadow prices) for all constraints."""
        pass

    @abstractmethod
    def write_solution(self, path: str | Path) -> None:
        """Write solution to file."""
        pass
