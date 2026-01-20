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
        """Export model to LP format."""
        if use_rust:
            try:
                from .writers.lp_rust import write_lp_rust

                write_lp_rust(self, filename)
                return
            except ImportError:
                pass
        from .writers.lp import write_lp

        write_lp(self, filename)

    def __repr__(self):
        n_vars = sum(v.size for v in self.variables.values())
        return f"Model('{self.name}', {n_vars} vars, {self.n_constraints} cons)"
