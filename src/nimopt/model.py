"""
Model class with nimblend-optimized constraint handling.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .expression import Constraint, LinearExpr
from .sets import Set
from .variable import Variable, VarRef


class Model:
    """
    Optimization model with nimblend backend.

    Constraints are stored symbolically and expanded lazily during
    LP generation.
    """

    def __init__(self, name: str = "model", sense: str = "minimize"):
        self.name = name
        self.sense = sense.lower()
        if self.sense not in ("minimize", "maximize"):
            raise ValueError("sense must be 'minimize' or 'maximize'")

        self.variables: Dict[str, Variable] = {}
        self._constraints: Dict[str, Constraint] = {}
        self.objective: Optional[LinearExpr] = None

    def var(
        self,
        name: str,
        sets: Optional[List[Set]] = None,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        vtype: str = "continuous",
    ) -> Variable:
        """Create a decision variable."""
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists")
        v = Variable(name, sets, lb, ub, vtype, model=self)
        self.variables[name] = v
        return v

    def eq(self, name: str, constraint: Constraint) -> None:
        """Add constraint(s) to the model."""
        if name in self._constraints:
            raise ValueError(f"Constraint '{name}' already exists")
        constraint.validate(name)
        self._constraints[name] = constraint

    def set_objective(self, expr) -> None:
        """Set the objective function."""
        if isinstance(expr, Variable):
            self.objective = LinearExpr.from_term(VarRef(expr, ()), 1.0)
        elif isinstance(expr, VarRef):
            self.objective = LinearExpr.from_term(expr, 1.0)
        elif isinstance(expr, LinearExpr):
            self.objective = expr
        elif isinstance(expr, Constraint):
            if expr.sense != "==":
                raise ValueError("Objective constraint must use '=='")
            if isinstance(expr.rhs, LinearExpr):
                self.objective = expr.rhs
        else:
            raise TypeError(f"Cannot set objective to {type(expr)}")

    @property
    def n_constraints(self) -> int:
        """Total number of scalar constraints."""
        total = 0
        for con in self._constraints.values():
            free = con.free_sets
            if not free:
                total += 1
            else:
                n = 1
                for s in free:
                    n *= len(s)
                total += n
        return total

    def to_lp(self, filename: str, use_rust: bool = True) -> None:
        """Export model to LP format.

        use_rust=True (default) requires the nimopt_rust extension and
        raises ImportError if it is not installed - there is no silent
        fallback. Pass use_rust=False explicitly to use the pure-Python
        reference writer.
        """
        if use_rust:
            try:
                from .writers.lp_rust import write_lp_rust
            except ImportError as e:
                raise ImportError(_RUST_MISSING_MSG) from e
            write_lp_rust(self, filename)
            return
        from .writers.lp import write_lp

        write_lp(self, filename)

    def solve(
        self,
        time_limit: Optional[float] = None,
        gap: Optional[float] = None,
        options: Optional[Dict] = None,
    ):
        """Solve the model with the direct HiGHS interface (no LP file).

        The problem is built in memory as sparse matrices by nimopt_rust
        and passed straight to HiGHS; columns follow variable insertion
        order by construction. Requires nimopt_rust (raises ImportError
        otherwise).

        Returns:
            (SolverResult, Solution | None) - Solution is None unless the
            model solved to optimality.
        """
        from .solvers.base import SolverStatus
        from .solvers.highs_direct import HiGHSDirectSolver

        solver = HiGHSDirectSolver()
        if time_limit is not None:
            solver.set_option("time_limit", time_limit)
        if gap is not None:
            solver.set_option("mip_rel_gap", gap)
        for k, v in (options or {}).items():
            solver.set_option(k, v)

        solver.load_model(self)
        result = solver.solve()
        solution = (
            solver.get_solution() if result.status == SolverStatus.OPTIMAL else None
        )
        return result, solution

    def __repr__(self):
        n_vars = sum(v.size for v in self.variables.values())
        return f"Model('{self.name}', {n_vars} vars, {self.n_constraints} cons)"


_RUST_MISSING_MSG = (
    "nimopt_rust extension is not installed. nimopt requires its Rust core "
    "for LP writing and direct solving (no Python fallback). Install a "
    "prebuilt wheel (pip install nimopt-rust) or build it locally with: "
    "cd rust && maturin build --release && pip install target/wheels/*.whl"
)
