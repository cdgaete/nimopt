"""GAMS/CPLEX solver wrapper."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .base import Solver, SolverResult, SolverStatus

if TYPE_CHECKING:
    from ..model import Model
    from ..solution import Solution

GAMS_DIR = "/opt/gams/gams46.4_linux_x64_64_sfx"


class GamsCplexSolver(Solver):
    """Wrapper for GAMS with CPLEX solver.

    Uses GAMS to solve LP/MIP problems with CPLEX backend.
    Requires GAMS installation with CPLEX license.
    """

    def __init__(self, gams_dir: str = GAMS_DIR):
        self.gams_dir = Path(gams_dir)
        if not self.gams_dir.exists():
            raise FileNotFoundError(f"GAMS directory not found: {gams_dir}")

        self._gams_exe = self.gams_dir / "gams"
        self._mps2gms = self.gams_dir / "mps2gms"

        self._lp_file: Optional[Path] = None
        self._work_dir: Optional[Path] = None
        self._result: Optional[SolverResult] = None
        self._model: Optional["Model"] = None
        self._var_names: list[str] = []
        self._var_values: list[float] = []
        self._var_duals: list[float] = []
        self._con_names: list[str] = []
        self._con_duals: list[float] = []

    def set_model(self, model: "Model") -> None:
        """Store model reference for solution extraction."""
        self._model = model

    def read_lp(self, path: str | Path) -> None:
        """Load model from LP file."""
        self._lp_file = Path(path)
        if not self._lp_file.exists():
            raise FileNotFoundError(f"LP file not found: {path}")

    def read_mps(self, path: str | Path) -> None:
        """Load model from MPS file."""
        self._lp_file = Path(path)
        if not self._lp_file.exists():
            raise FileNotFoundError(f"MPS file not found: {path}")

    def set_option(self, name: str, value) -> None:
        """Set solver option (stored for CPLEX options file)."""
        # TODO: implement options
        pass

    def solve(self) -> SolverResult:
        """Solve using GAMS/CPLEX."""
        if self._lp_file is None:
            raise RuntimeError("No model loaded. Call read_lp() first.")

        # Create temp work directory
        self._work_dir = Path(tempfile.mkdtemp(prefix="nimopt_gams_"))

        try:
            # Convert LP to GDX + GMS
            gdx_file = self._work_dir / "model.gdx"
            gms_file = self._work_dir / "model.gms"

            env = os.environ.copy()
            env["PATH"] = f"{self.gams_dir}:{env.get('PATH', '')}"

            subprocess.run(
                [str(self._mps2gms), str(self._lp_file), str(gdx_file), str(gms_file)],
                env=env,
                capture_output=True,
                check=True,
            )

            # Modify GMS to output solution to GDX
            self._add_solution_output(gms_file)

            # Run GAMS with CPLEX
            subprocess.run(
                [
                    str(self._gams_exe),
                    str(gms_file),
                    "lp=cplex",
                    "mip=cplex",
                    "lo=2",
                    f"curdir={self._work_dir}",
                ],
                env=env,
                capture_output=True,
                cwd=self._work_dir,
            )

            # Parse results
            self._parse_solution()

            return self._result

        except subprocess.CalledProcessError:
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                objective_value=None,
                solve_time=0,
            )

    def _add_solution_output(self, gms_file: Path) -> None:
        """Add solution export to GMS file."""
        with open(gms_file, "a") as f:
            f.write("""
* Export solution to GDX
execute_unload 'solution.gdx', xc, obj, eg, el, ee;
""")

    def _parse_solution(self) -> None:
        """Parse solution from GAMS output."""
        import gams

        sol_gdx = self._work_dir / "solution.gdx"
        lst_file = self._work_dir / "model.lst"

        # Parse objective and status from listing file
        obj_value = None
        status = SolverStatus.UNKNOWN
        solve_time = 0.0

        if lst_file.exists():
            with open(lst_file) as f:
                content = f.read()
                if "Optimal" in content or "optimal" in content:
                    status = SolverStatus.OPTIMAL
                elif "Infeasible" in content:
                    status = SolverStatus.INFEASIBLE
                elif "Unbounded" in content:
                    status = SolverStatus.UNBOUNDED

                # Extract objective value (LP format)
                for line in content.split("\n"):
                    if "Objective:" in line:
                        try:
                            obj_value = float(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    # MIP format
                    elif "OBJECTIVE VALUE" in line:
                        try:
                            obj_value = float(line.split()[-1])
                        except (ValueError, IndexError):
                            pass

        # Read solution from GDX
        if sol_gdx.exists():
            ws = gams.GamsWorkspace(
                system_directory=str(self.gams_dir),
                working_directory=str(self._work_dir),
            )
            db = ws.add_database_from_gdx(str(sol_gdx))

            # Get variable values
            try:
                xc = db.get_variable("xc")
                for rec in xc:
                    self._var_names.append(rec.key(0))
                    self._var_values.append(rec.level)
                    self._var_duals.append(rec.marginal)
            except gams.GamsException:
                pass

            # Get constraint duals (from eg, el, ee equations)
            # Collect all constraints
            con_dict = {}
            for eq_name in ["eg", "el", "ee"]:
                try:
                    eq = db.get_equation(eq_name)
                    for rec in eq:
                        con_dict[rec.key(0)] = rec.marginal
                except gams.GamsException:
                    pass

            # Order constraints to match LP file order
            # Read LP file to get constraint order
            if self._lp_file and self._lp_file.exists():
                with open(self._lp_file) as f:
                    in_constraints = False
                    for line in f:
                        line = line.strip()
                        if line.startswith("Subject To"):
                            in_constraints = True
                            continue
                        if in_constraints:
                            if line.startswith("Bounds") or not line:
                                break
                            # Extract constraint name (before the colon)
                            if ":" in line:
                                con_name = line.split(":")[0].strip()
                                if con_name in con_dict:
                                    self._con_names.append(con_name)
                                    self._con_duals.append(con_dict[con_name])

        self._result = SolverResult(
            status=status,
            objective_value=obj_value,
            solve_time=solve_time,
        )

    def get_variable_names(self) -> list[str]:
        return self._var_names

    def get_constraint_names(self) -> list[str]:
        return self._con_names

    def get_variable_values(self) -> list[float]:
        return self._var_values

    def get_variable_duals(self) -> list[float]:
        return self._var_duals

    def get_constraint_duals(self) -> list[float]:
        return self._con_duals

    def get_objective_value(self) -> float:
        return self._result.objective_value if self._result else None

    def write_solution(self, path: str | Path) -> None:
        """Write solution to file."""
        # Not implemented - GAMS handles solution internally
        pass

    def get_solution(self) -> "Solution":
        """Extract solution as nimblend Arrays.

        Requires set_model() to be called first.
        """
        from ..solution import extract_solution_python

        if self._model is None:
            raise RuntimeError("No model set. Call set_model(model) first.")
        return extract_solution_python(self, self._model)
