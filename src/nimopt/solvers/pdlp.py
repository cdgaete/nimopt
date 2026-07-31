"""PDLP solver interface - first-order LP solver.

Supports two backends:
1. OR-Tools PDLP (CPU) - always available
2. cuPDLPx (GPU) - when CUDA is available

PDLP uses matrix-vector multiplication rather than factorization,
making it efficient for large-scale LPs and GPU-friendly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import scipy.sparse as sp

from .base import Solver, SolverResult, SolverStatus

if TYPE_CHECKING:
    from ..model import Model

# Check available backends
HAS_ORTOOLS = False
try:
    from ortools.linear_solver import pywraplp
    # Verify PDLP is actually available
    _test = pywraplp.Solver.CreateSolver('PDLP')
    HAS_ORTOOLS = _test is not None
    del _test
except ImportError:
    pass

HAS_CUPDLP = False
try:
    from cupdlpx import Model as CuModel, PDLP as CuPDLP
    HAS_CUPDLP = True
except ImportError:
    pass

HAS_RUST = False
try:
    import nimopt_rust
    HAS_RUST = True
except ImportError:
    pass


class PDLPSolver(Solver):
    """PDLP first-order LP solver with CPU/GPU backends.
    
    Parameters
    ----------
    backend : str, optional
        'auto' (default), 'ortools', or 'cupdlp'
    use_rust : bool, optional
        Use Rust for matrix building (default: True)
    time_limit : float, optional
        Time limit in seconds (default: 3600)
    tolerance : float, optional
        Optimality tolerance (default: 1e-6)
    verbose : bool, optional
        Print solver output (default: False)
    """

    def __init__(
        self,
        backend: str = 'auto',
        use_rust: bool = True,
        time_limit: float = 3600.0,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        self._backend = self._select_backend(backend)
        if use_rust and not HAS_RUST:
            from ..model import _RUST_MISSING_MSG

            raise ImportError(_RUST_MISSING_MSG)
        self._use_rust = use_rust
        self._time_limit = time_limit
        self._tolerance = tolerance
        self._verbose = verbose
        
        self._model: Optional[Model] = None
        self._matrices: Optional[dict] = None
        self._result: Optional[SolverResult] = None
        
        # Solution storage
        self._primal: Optional[np.ndarray] = None
        self._dual: Optional[np.ndarray] = None
        self._var_names: Optional[List[str]] = None
        self._con_names: Optional[List[str]] = None

    def _select_backend(self, backend: str) -> str:
        if backend == 'auto':
            if HAS_CUPDLP:
                return 'cupdlp'
            elif HAS_ORTOOLS:
                return 'ortools'
            else:
                raise ImportError("No PDLP backend available. Install ortools or cupdlpx.")
        if backend == 'cupdlp' and not HAS_CUPDLP:
            raise ImportError("cupdlpx not installed. Install with: pip install cupdlpx")
        if backend == 'ortools' and not HAS_ORTOOLS:
            raise ImportError("ortools not installed or PDLP not available")
        return backend

    def read_lp(self, path) -> None:
        raise NotImplementedError("PDLP solver uses direct model loading, not LP files")

    def read_mps(self, path) -> None:
        raise NotImplementedError("PDLP solver uses direct model loading, not MPS files")

    def set_option(self, name: str, value) -> None:
        if name == 'time_limit':
            self._time_limit = float(value)
        elif name == 'tolerance':
            self._tolerance = float(value)
        elif name == 'verbose':
            self._verbose = bool(value)

    def load_model(self, model: "Model") -> None:
        """Build matrices from nimopt Model."""
        from .highs_direct import _build_matrices_rust, _build_matrices_python
        
        self._model = model
        if self._use_rust:
            self._matrices = _build_matrices_rust(model)
        else:
            self._matrices = _build_matrices_python(model)
        
        self._var_names = self._matrices.get('var_names')
        self._con_names = self._matrices.get('con_names')

    def solve(self) -> SolverResult:
        """Solve the loaded model."""
        if self._matrices is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        if self._backend == 'cupdlp':
            return self._solve_cupdlp()
        else:
            return self._solve_ortools()

    def _solve_ortools(self) -> SolverResult:
        """Solve using OR-Tools PDLP."""
        import time
        from ortools.linear_solver import pywraplp
        
        m = self._matrices
        n_vars = m['n_vars']
        n_cons = m['n_cons']
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('PDLP')
        if solver is None:
            raise RuntimeError("Could not create PDLP solver")
        
        # Set parameters
        solver.SetTimeLimit(int(self._time_limit * 1000))  # ms
        
        # Create variables
        vars_list = []
        for i in range(n_vars):
            lb_val = m['lb'][i]
            ub_val = m['ub'][i]
            lb = lb_val if np.isfinite(lb_val) else -solver.infinity()
            ub = ub_val if np.isfinite(ub_val) else solver.infinity()
            vars_list.append(solver.NumVar(lb, ub, f'x{i}'))
        
        # Set objective
        obj = solver.Objective()
        for i, c in enumerate(m['c']):
            if c != 0:
                obj.SetCoefficient(vars_list[i], c)
        
        if self._model and self._model.sense == 'maximize':
            obj.SetMaximization()
        else:
            obj.SetMinimization()
        
        # Add constraints using CSR data
        indptr = m['indptr']
        indices = m['indices']
        data = m['data']
        row_lower = m['row_lower']
        row_upper = m['row_upper']
        
        for row in range(n_cons):
            start, end = indptr[row], indptr[row + 1]
            lb_val = row_lower[row]
            ub_val = row_upper[row]
            lb = lb_val if np.isfinite(lb_val) else -solver.infinity()
            ub = ub_val if np.isfinite(ub_val) else solver.infinity()
            
            ct = solver.Constraint(lb, ub)
            for j in range(start, end):
                ct.SetCoefficient(vars_list[indices[j]], data[j])
        
        # Solve
        start_time = time.time()
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        # Map status
        status_map = {
            pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
            pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
            pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
        }
        result_status = status_map.get(status, SolverStatus.UNKNOWN)
        
        # Extract solution
        if status == pywraplp.Solver.OPTIMAL:
            self._primal = np.array([v.solution_value() for v in vars_list])
            self._dual = np.zeros(n_cons)  # OR-Tools PDLP doesn't expose duals easily
        
        self._result = SolverResult(
            status=result_status,
            objective_value=solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else None,
            solve_time=solve_time,
            iterations=solver.iterations(),
        )
        return self._result

    def _solve_cupdlp(self) -> SolverResult:
        """Solve using cuPDLPx (GPU).
        
        NOTE: cuPDLPx has a bug where variable bounds (lb, ub) are not properly
        enforced - the solver satisfies constraint bounds but ignores variable
        bounds, leading to incorrect solutions. As a workaround, we convert
        finite variable bounds to explicit constraint rows.
        """
        from cupdlpx import Model as CuModel, PDLP as CuPDLP
        
        m = self._matrices
        n_vars = m['n_vars']
        n_cons = m['n_cons']
        
        # Build sparse constraint matrix in CSR format
        orig_A = sp.csr_matrix(
            (m['data'], m['indices'], m['indptr']),
            shape=(n_cons, n_vars)
        )
        
        # WORKAROUND: Convert variable bounds to explicit constraints
        # cuPDLPx doesn't properly enforce variable bounds, only constraint bounds
        lb = m['lb']
        ub = m['ub']
        
        # Find finite lower and upper bounds
        finite_lb_mask = lb > -np.inf
        finite_ub_mask = ub < np.inf
        n_lb = np.sum(finite_lb_mask)
        n_ub = np.sum(finite_ub_mask)
        
        if n_lb > 0 or n_ub > 0:
            # Build bound constraint matrices
            bound_rows = []
            bound_row_lower = []
            bound_row_upper = []
            
            if n_lb > 0:
                # x_i >= lb_i  =>  row with coef 1, bounds [lb_i, inf]
                lb_indices = np.where(finite_lb_mask)[0].astype(np.int32)
                lb_data = np.ones(n_lb, dtype=np.float64)
                lb_indptr = np.arange(n_lb + 1, dtype=np.int32)
                lb_A = sp.csr_matrix((lb_data, lb_indices, lb_indptr), shape=(n_lb, n_vars))
                bound_rows.append(lb_A)
                bound_row_lower.append(lb[finite_lb_mask])
                bound_row_upper.append(np.full(n_lb, np.inf))
            
            if n_ub > 0:
                # x_i <= ub_i  =>  row with coef 1, bounds [-inf, ub_i]
                ub_indices = np.where(finite_ub_mask)[0].astype(np.int32)
                ub_data = np.ones(n_ub, dtype=np.float64)
                ub_indptr = np.arange(n_ub + 1, dtype=np.int32)
                ub_A = sp.csr_matrix((ub_data, ub_indices, ub_indptr), shape=(n_ub, n_vars))
                bound_rows.append(ub_A)
                bound_row_lower.append(np.full(n_ub, -np.inf))
                bound_row_upper.append(ub[finite_ub_mask])
            
            # Stack all constraints
            A = sp.vstack([orig_A] + bound_rows, format='csr')
            row_lower = np.concatenate([m['row_lower']] + bound_row_lower)
            row_upper = np.concatenate([m['row_upper']] + bound_row_upper)
            
            # Use free variable bounds (handled by explicit constraints)
            var_lb = np.full(n_vars, -np.inf)
            var_ub = np.full(n_vars, np.inf)
        else:
            A = orig_A
            row_lower = m['row_lower']
            row_upper = m['row_upper']
            var_lb = lb
            var_ub = ub
        
        # cuPDLPx uses constraint bounds format
        model = CuModel(
            objective_vector=m['c'],
            constraint_matrix=A,
            constraint_lower_bound=row_lower,
            constraint_upper_bound=row_upper,
            variable_lower_bound=var_lb,
            variable_upper_bound=var_ub,
        )
        
        # Set sense
        if self._model and self._model.sense == 'maximize':
            model.ModelSense = CuPDLP.MAXIMIZE
        
        # Set parameters
        model.setParam('TimeLimit', self._time_limit)
        model.setParam('OptimalityTol', self._tolerance)
        model.setParam('FeasibilityTol', self._tolerance)
        model.setParam('OutputFlag', self._verbose)
        
        # Solve
        model.optimize()
        
        # Map status
        status_map = {
            'OPTIMAL': SolverStatus.OPTIMAL,
            'INFEASIBLE': SolverStatus.INFEASIBLE,
            'UNBOUNDED': SolverStatus.UNBOUNDED,
            'TIME_LIMIT': SolverStatus.TIME_LIMIT,
            'ITERATION_LIMIT': SolverStatus.ITERATION_LIMIT,
        }
        result_status = status_map.get(model.Status, SolverStatus.UNKNOWN)
        
        # Extract solution
        if model.X is not None:
            self._primal = np.array(model.X)
        if model.Pi is not None:
            self._dual = np.array(model.Pi)
        
        self._result = SolverResult(
            status=result_status,
            objective_value=model.ObjVal,
            solve_time=model.Runtime,
            iterations=model.IterCount,
            gap=model.RelGap,
        )
        return self._result

    def get_variable_names(self) -> List[str]:
        if self._var_names is None and self._matrices:
            from .highs_direct import _generate_var_names_from_info
            var_info = self._matrices.get('var_info')
            if var_info:
                self._var_names = _generate_var_names_from_info(var_info)
        return self._var_names or []

    def get_constraint_names(self) -> List[str]:
        return self._con_names or []

    def get_variable_values(self) -> List[float]:
        return list(self._primal) if self._primal is not None else []

    def get_variable_duals(self) -> List[float]:
        # Reduced costs not directly available from PDLP
        return [0.0] * len(self._primal) if self._primal is not None else []

    def get_constraint_duals(self) -> List[float]:
        return list(self._dual) if self._dual is not None else []

    def write_solution(self, path) -> None:
        if self._primal is None:
            raise RuntimeError("No solution available")
        
        with open(path, 'w') as f:
            f.write(f"# Objective: {self._result.objective_value}\n")
            f.write(f"# Status: {self._result.status.value}\n")
            f.write(f"# Solve time: {self._result.solve_time:.3f}s\n")
            f.write(f"# Iterations: {self._result.iterations}\n\n")
            
            names = self.get_variable_names()
            for i, val in enumerate(self._primal):
                name = names[i] if i < len(names) else f'x{i}'
                f.write(f"{name} = {val}\n")

    def get_solution(self):
        """Extract solution as nimblend Arrays."""
        from ..solution import extract_solution_python
        
        if self._model is None:
            raise RuntimeError("No model loaded")
        return extract_solution_python(self, self._model)

    @property
    def backend(self) -> str:
        return self._backend
