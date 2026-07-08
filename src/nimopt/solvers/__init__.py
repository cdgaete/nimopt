"""Solver abstractions for nimopt.

Solver classes are imported lazily so that optional dependencies
(scipy for PDLP, GAMS for CPLEX) are only required when the
corresponding solver is actually used.
"""

from .base import Solver, SolverResult, SolverStatus

_LAZY = {
    "HiGHSSolver": ".highs",
    "HiGHSDirectSolver": ".highs_direct",
    "GamsCplexSolver": ".gams_cplex",
    "PDLPSolver": ".pdlp",
}

__all__ = ["Solver", "SolverStatus", "SolverResult", *_LAZY.keys()]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
