"""The GAMS adapter.

The adapter solves with CPLEX through the expert-level API of GAMS. GMO stores
the model and GEV calls the solver, both in this process. The adapter loads
both from the GAMS library `libjoatdclib64.so` with `ctypes`. It requires no
GAMS Python package. The GAMS system directory is the directory of the `gams`
executable on PATH.

GEV reads the license and the environment from a control file. A GAMS run on a
one-row model writes that file once per process and system directory.

GMO stores the objective as a free column `z` and the equality row
`z - c'x = 0`. The adapter adds one column and one row per call. It frees GMO
and GEV after it reads the values, and `Result.backend` is None.

A solve reports a model status and a solve status. `OUTCOME` maps a model
status at normal completion into `STATUS`. `LIMITS` maps a solve status into
`STATUS`. At the solve status `terminated by solver`, the iteration count and
the node count identify the limit. `FAILED` lists the solve statuses this
adapter raises on. A status in none of them raises.

The options file sets `rerun yes`. Where presolve reports a model infeasible or
unbounded, CPLEX then solves it again without presolve.

The option `separate_process` runs CPLEX in a process of its own. GEV then
writes the model, the solution and the log to the scratch directory. The
adapter writes the log to standard output after the solve and removes the
files.
"""

import atexit
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.assembly import Assembled

from nimopt.row import senses
from nimopt.solvers.base import (
    Capabilities,
    Result,
    check_reported,
    check_sense,
    proved_bound,
)
from nimopt.solvers.options import translated

BACKEND = "libjoatdclib64"

LIBRARY = {
    "linux": "libjoatdclib64.so",
    "darwin": "libjoatdclib64.dylib",
    "win32": "joatdclib64.dll",
}

CAPABILITIES = Capabilities(
    "gams",
    {
        "integrality": "native",
        "duals": "native",
        "conflict": "absent",
        "ray": "absent",
    },
    rejected=(("integrality", "duals"),),
)

MODEL_STATUS = {
    1: "optimal",
    2: "locally optimal",
    3: "unbounded",
    4: "infeasible",
    5: "locally infeasible",
    6: "intermediate infeasible",
    7: "feasible solution",
    8: "integer solution",
    9: "intermediate non-integer",
    10: "integer infeasible",
    11: "licensing problem",
    12: "error unknown",
    13: "error no solution",
    14: "no solution returned",
    15: "solved unique",
    16: "solved",
    17: "solved singular",
    18: "unbounded - no solution",
    19: "infeasible - no solution",
}

SOLVE_STATUS = {
    1: "normal completion",
    2: "iteration interrupt",
    3: "resource interrupt",
    4: "terminated by solver",
    5: "evaluation interrupt",
    6: "capability problems",
    7: "licensing problems",
    8: "user interrupt",
    9: "setup failure",
    10: "solver failure",
    11: "internal solver failure",
    12: "solve processing skipped",
    13: "system failure",
}

OUTCOME = {
    "optimal": "optimal",
    "integer solution": "optimal",
    "unbounded": "unbounded",
    "unbounded - no solution": "unbounded",
    "infeasible": "infeasible",
    "integer infeasible": "infeasible",
    "infeasible - no solution": "unbounded_or_infeasible",
}

FEASIBLE = ("optimal", "locally optimal", "feasible solution", "integer solution")

LIMITS = {
    "iteration interrupt": "iteration_limit",
    "resource interrupt": "time_limit",
    "user interrupt": "interrupt",
}

NORMAL = "normal completion"

TERMINATED = "terminated by solver"

LICENSING = "licensing problems"

FAILED = (
    "evaluation interrupt",
    "capability problems",
    LICENSING,
    "setup failure",
    "solver failure",
    "internal solver failure",
    "solve processing skipped",
    "system failure",
)

OPTION_NAMES = {
    "time_limit": "reslim",
    "iteration_limit": "iterlim",
    "node_limit": "nodelim",
    "mip_gap": "optcr",
    "mip_abs_gap": "optca",
    "feasibility_tol": "eprhs",
    "optimality_tol": "epopt",
    "threads": "threads",
    "seed": "randomseed",
    "log": "logoption",
    "presolve": "preind",
    "method": "lpmethod",
    "newton_system": None,
    "crossover": "solutiontype",
    "pdlp_tol": None,
    "separate_process": "solvelink",
}

LOAD_LIBRARY, CALL_MODULE = 5, 2

OPTION_VALUES: dict[str, dict[Any, int]] = {
    "presolve": {"off": 0, "choose": 1, "on": 1},
    "method": {"choose": 0, "simplex": 2, "barrier": 4},
    "crossover": {"choose": 0, "off": 2, "on": 1},
    "separate_process": {False: LOAD_LIBRARY, True: CALL_MODULE},
}

CALL_OPTIONS = {
    "reslim": 1e10,
    "iterlim": 2_147_483_647,
    "optcr": 1e-4,
    "optca": 0.0,
    "logoption": False,
    "solvelink": LOAD_LIBRARY,
}

FIXED_OPTIONS = {"rerun": "yes"}

API_VERSIONS = {"gmo": 26, "gev": 7}

CONTROL_MODEL = (
    "Variable z; Equation e; e.. z =e= 0; Model m /e/;\n"
    "m.justScrDir = 1; solve m using lp minimizing z;\n"
)

MESSAGE = 256

CHUNK = 65536

EQUATION = {"==": 0, ">=": 1, "<=": 2}

CONTINUOUS, INTEGER = 0, 2

SUPERBASIC = 3

LP, MIP = 1, 2

SENSES = {"min": 0, "max": 1}

ITERATIONS, OBJECTIVE, NODES, MIP_BOUND = 3, 5, 11, 15

SAME_STREAMS, QUIET, OWN_FILE = 0, 1, 2

_P, _I, _D, _S = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_char_p

ENTRIES = {
    "gmoInitData": ("gmoinitdata", _I, (_P, _I, _I, _I), (3, 3, 3, 3)),
    "gmoAddRow": (
        "gmoaddrow",
        _I,
        (_P, _I, _I, _D, _D, _D, _D, _I, _I, _P, _P, _P),
        (3, 3, 3, 13, 13, 13, 13, 3, 3, 7, 5, 7),
    ),
    "gmoAddCol": (
        "gmoaddcol",
        _I,
        (_P, _I, _D, _D, _D, _D, _I, _I, _D, _D, _I, _P, _P, _P),
        (3, 3, 13, 13, 13, 13, 3, 3, 13, 13, 3, 7, 5, 7),
    ),
    "gmoCompleteData": ("cgmocompletedata", _I, (_P, _S), (3, 12)),
    "gmoRegisterEnvironment": ("cgmoregisterenvironment", _I, (_P, _P, _S), (3, 1, 12)),
    "gmoModelTypeSet": ("gmomodeltypeset", None, (_P, _I), (0, 3)),
    "gmoSenseSet": ("gmosenseset", None, (_P, _I), (0, 3)),
    "gmoObjVarSet": ("gmoobjvarset", None, (_P, _I), (0, 3)),
    "gmoPinf": ("gmopinf", _D, (_P,), (13,)),
    "gmoMinf": ("gmominf", _D, (_P,), (13,)),
    "gmoOptFileSet": ("gmooptfileset", None, (_P, _I), (0, 3)),
    "gmoNameOptFileSet": ("cgmonameoptfileset", None, (_P, _S), (0, 12)),
    "gmoModelStat": ("gmomodelstat", _I, (_P,), (3,)),
    "gmoSolveStat": ("gmosolvestat", _I, (_P,), (3,)),
    "gmoGetHeadnTail": ("gmogetheadntail", _D, (_P, _I), (13, 3)),
    "gmoGetVarL": ("gmogetvarl", _I, (_P, _P), (3, 6)),
    "gmoGetEquM": ("gmogetequm", _I, (_P, _P), (3, 6)),
    "gevInitEnvironmentLegacy": ("cgevinitenvironmentlegacy", _I, (_P, _S), (3, 11)),
    "gevCallSolver": (
        "cgevcallsolver",
        _I,
        (_P, _P, _S, _S, _I, _I, _S, _S, _D, _I, _I, _D, _D, _P, _S),
        (3, 1, 11, 11, 3, 3, 11, 11, 13, 3, 3, 13, 13, 2, 12),
    ),
}

HANDLES = ("gmoxcreate", "gmoxfree", "gevxcreate", "gevxfree")


def _library_name(system: str | None = None) -> str:
    """Return the file name of the GAMS library on `system`, by default this one.

    `system` is a value of `sys.platform`. Raises RuntimeError for a platform
    outside `LIBRARY`.
    """
    system = sys.platform if system is None else system
    if system not in LIBRARY:
        raise RuntimeError(
            f"the gams adapter supports the platforms {tuple(LIBRARY)}; this "
            f"platform is {system!r}; solve with another solver"
        )
    return LIBRARY[system]


def _executable(directory: Path) -> Path:
    """Return the `gams` executable in `directory`.

    Raises RuntimeError where `directory` contains no `gams` executable.
    """
    found = shutil.which("gams", path=str(directory))
    if found is None:
        raise RuntimeError(
            f"{directory} contains no `gams` executable; add a GAMS system "
            f"directory to PATH"
        )
    return Path(found)


def system_directory() -> Path:
    """Return the GAMS system directory: the directory of `gams` on PATH.

    Raises RuntimeError where PATH contains no `gams` executable, and where the
    directory of `gams` contains no GAMS library.
    """
    found = shutil.which("gams")
    if found is None:
        raise RuntimeError(
            "no `gams` executable on PATH; add the GAMS system directory to PATH"
        )
    directory = Path(found).resolve().parent
    if not (directory / _library_name()).is_file():
        raise RuntimeError(
            f"{directory} contains `gams` and no {_library_name()}; add a GAMS "
            f"system directory to PATH"
        )
    return directory


def installed() -> bool:
    """Return whether PATH contains a GAMS system directory with its library."""
    try:
        system_directory()
    except RuntimeError:
        return False
    return True


def _check_entries(confirms: Callable[[str, tuple[int, ...]], bool]) -> None:
    """Raise RuntimeError for an entry point whose signature `confirms` rejects.

    `confirms` receives the name of an entry point and the type codes of its
    return value and its arguments after the handle.
    """
    for name, (_, _, _, codes) in ENTRIES.items():
        if not confirms(name, codes):
            raise RuntimeError(
                f"the GAMS library has another signature for {name}; install "
                f"a GAMS release with GMO API {API_VERSIONS['gmo']} and GEV API "
                f"{API_VERSIONS['gev']}"
            )


class _Library:
    """The GMO and GEV entry points of one GAMS system directory."""

    gmoInitData: Any
    gmoAddRow: Any
    gmoAddCol: Any
    gmoCompleteData: Any
    gmoRegisterEnvironment: Any
    gmoModelTypeSet: Any
    gmoSenseSet: Any
    gmoObjVarSet: Any
    gmoPinf: Any
    gmoMinf: Any
    gmoOptFileSet: Any
    gmoNameOptFileSet: Any
    gmoModelStat: Any
    gmoSolveStat: Any
    gmoGetHeadnTail: Any
    gmoGetVarL: Any
    gmoGetEquM: Any
    gevInitEnvironmentLegacy: Any
    gevCallSolver: Any
    gmoxcreate: Any
    gmoxfree: Any
    gevxcreate: Any
    gevxfree: Any

    def __init__(self, directory: Path) -> None:
        if sys.platform == "win32":
            self.search = os.add_dll_directory(str(directory))
        try:
            dll = ctypes.CDLL(str(directory / _library_name()))
        except OSError as error:
            raise RuntimeError(
                f"the GAMS library in {directory} does not load into this "
                f"Python: {error}; install a GAMS build for {platform.machine()}"
            ) from error
        message = ctypes.create_string_buffer(MESSAGE)
        compatible = ctypes.c_int()
        checks = {}
        for prefix, version in API_VERSIONS.items():
            supports = getattr(dll, f"c{prefix}xapiversion")
            supports.restype = _I
            supports.argtypes = (_I, _S, ctypes.POINTER(_I))
            if not supports(version, message, ctypes.byref(compatible)):
                raise RuntimeError(
                    f"the GAMS library at {directory} does not support "
                    f"{prefix.upper()} API {version}: "
                    f"{_text(message)}; install a GAMS "
                    f"release that supports it"
                )
            check = getattr(dll, f"c{prefix}xcheck")
            check.restype = _I
            check.argtypes = (_S, _I, ctypes.POINTER(_I), _S)
            checks[prefix] = check

        def confirms(name: str, codes: tuple[int, ...]) -> bool:
            array = (_I * len(codes))(*codes)
            return bool(checks[name[:3]](name.encode(), len(codes) - 1, array, message))

        _check_entries(confirms)
        for name, (symbol, restype, argtypes, _) in ENTRIES.items():
            entry = getattr(dll, symbol)
            entry.restype = restype
            entry.argtypes = argtypes
            setattr(self, name, entry)
        for symbol in HANDLES:
            entry = getattr(dll, symbol)
            entry.restype = None
            entry.argtypes = (ctypes.POINTER(_P),)
            setattr(self, symbol, entry)


@cache
def _library(directory: Path) -> _Library:
    """Return the checked entry points of the GAMS library in `directory`."""
    return _Library(directory)


@cache
def _control_file(directory: Path) -> Path:
    """Return the control file a GAMS run in `directory` writes for GEV.

    The file and its directory are removed when the process exits. Raises
    RuntimeError where GAMS returns a nonzero code.
    """
    work = Path(tempfile.mkdtemp(prefix="nimopt-gams-"))
    atexit.register(shutil.rmtree, work, True)
    scratch = work / "scratch"
    scratch.mkdir()
    (work / "control.gms").write_text(CONTROL_MODEL)
    run = subprocess.run(
        [
            str(_executable(directory)),
            "control.gms",
            f"scrdir={scratch}",
            "solvercntr=gamscntr.dat",
            "lo=3",
            f"curdir={work}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(
            f"GAMS returned {run.returncode} writing its control file in "
            f"{work}: {run.stdout[-500:]}; run `gams` in a terminal to read the "
            f"license and the installation"
        )
    return scratch / "gamscntr.dat"


def _text(message: Any) -> str:
    """Return the text GMO or GEV wrote into a message buffer."""
    return message.value.decode(errors="replace")


def _address(array: npt.NDArray[Any]) -> int:
    """Return the address of the first element of a contiguous numpy array."""
    return int(array.__array_interface__["data"][0])


def _bounded(values: npt.NDArray[np.float64], plus: float, minus: float) -> Any:
    """Return `values` with each infinity replaced by the GMO infinity."""
    return np.where(
        np.isposinf(values), plus, np.where(np.isneginf(values), minus, values)
    )


def _load(library: _Library, model: Any, assembled: "Assembled", sense: str) -> None:
    """Load the assembled model into GMO, with the objective as column `z`.

    GMO requires the model type and the sense before `gmoInitData`, every
    column before the first row, and the objective column after the last row.
    Raises RuntimeError where GMO rejects a column, a row or the completed
    model.
    """
    message = ctypes.create_string_buffer(MESSAGE)
    n, m = assembled.n_cols, assembled.n_rows
    library.gmoModelTypeSet(model, MIP if assembled.integrality.any() else LP)
    library.gmoSenseSet(model, SENSES[sense])
    if library.gmoInitData(model, m + 1, n + 1, 0) != 0:
        raise RuntimeError(
            f"GMO rejected a model of {m} rows and {n} columns; report it as a defect"
        )
    plus, minus = library.gmoPinf(model), library.gmoMinf(model)

    lower = _bounded(assembled.col_lower, plus, minus)
    upper = _bounded(assembled.col_upper, plus, minus)
    level = np.clip(0.0, lower, upper)
    kinds = np.where(assembled.integrality.astype(bool), INTEGER, CONTINUOUS)
    indptr = np.asarray(assembled.indptr, dtype=np.int64)
    counts = np.diff(indptr)
    cost = np.asarray(assembled.col_cost, dtype=np.float64)
    used = np.flatnonzero(cost)
    width = max(int(counts.max(initial=0)), used.size + 1)
    flags = np.zeros(width, dtype=np.int32)
    empty = np.zeros(1, dtype=np.float64)
    at_flags, at_empty = _address(flags), _address(empty)

    add_col = library.gmoAddCol
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        for at, (kind, low, value, high) in enumerate(
            zip(
                kinds[start:stop].tolist(),
                lower[start:stop].tolist(),
                level[start:stop].tolist(),
                upper[start:stop].tolist(),
            ),
            start,
        ):
            if add_col(
                model,
                kind,
                low,
                value,
                high,
                0.0,
                SUPERBASIC,
                0,
                1.0,
                1.0,
                0,
                at_flags,
                at_empty,
                at_flags,
            ):
                raise RuntimeError(
                    f"GMO rejected column {at} with bounds [{low}, {high}]; "
                    f"report it as a defect"
                )
    if add_col(
        model,
        CONTINUOUS,
        minus,
        0.0,
        plus,
        0.0,
        SUPERBASIC,
        0,
        1.0,
        1.0,
        0,
        at_flags,
        at_empty,
        at_flags,
    ):
        raise RuntimeError("GMO rejected the objective column; report it as a defect")

    relations = senses(assembled.row_lower, assembled.row_upper)
    equations = np.select(
        [relations == "==", relations == "<="],
        [EQUATION["=="], EQUATION["<="]],
        EQUATION[">="],
    )
    right = _bounded(
        np.where(relations == ">=", assembled.row_lower, assembled.row_upper),
        plus,
        minus,
    )
    indices = np.ascontiguousarray(assembled.indices, dtype=np.int32)
    values = np.ascontiguousarray(assembled.values, dtype=np.float64)
    add_row = library.gmoAddRow
    for start in range(0, m, CHUNK):
        stop = min(start + CHUNK, m)
        first = indptr[start:stop]
        for at, (kind, rhs, count, columns, coefficients) in enumerate(
            zip(
                equations[start:stop].tolist(),
                right[start:stop].tolist(),
                counts[start:stop].tolist(),
                (_address(indices) + 4 * first).tolist(),
                (_address(values) + 8 * first).tolist(),
            ),
            start,
        ):
            if add_row(
                model,
                kind,
                0,
                0.0,
                1.0,
                rhs,
                0.0,
                SUPERBASIC,
                count,
                columns,
                coefficients,
                at_flags,
            ):
                raise RuntimeError(
                    f"GMO rejected row {at} with {count} entries; report it as a defect"
                )
    columns = np.append(used, n).astype(np.int32)
    coefficients = np.append(-cost[used], 1.0)
    if add_row(
        model,
        EQUATION["=="],
        0,
        0.0,
        1.0,
        0.0,
        0.0,
        SUPERBASIC,
        columns.size,
        _address(columns),
        _address(coefficients),
        at_flags,
    ):
        raise RuntimeError("GMO rejected the objective row; report it as a defect")
    library.gmoObjVarSet(model, n)
    if library.gmoCompleteData(model, message) != 0:
        raise RuntimeError(
            f"GMO rejected the completed model: {_text(message)}; report it as a defect"
        )


def _status(
    model_status: str,
    solve_status: str,
    limits: Mapping[str, int],
    iterations: float,
    nodes: float,
) -> str:
    """Return the `STATUS` member a model status and a solve status report.

    `limits` maps `iteration_limit` and `node_limit` to the values a caller
    set. Raises RuntimeError for a status this adapter maps to no outcome.
    """
    check_reported(
        solve_status,
        FAILED,
        (NORMAL, TERMINATED, *LIMITS),
        "CPLEX under GAMS",
        "solve status",
    )
    if solve_status in LIMITS:
        return LIMITS[solve_status]
    if solve_status == TERMINATED:
        counts = {"iteration_limit": iterations, "node_limit": nodes}
        reached = [name for name, limit in limits.items() if counts[name] >= limit]
        if not reached:
            raise RuntimeError(
                f"CPLEX under GAMS terminated at model status {model_status!r} and "
                f"reached no limit the options set; pass the option log=True to "
                f"read the cause in the CPLEX log"
            )
        if len(reached) > 1:
            raise RuntimeError(
                f"CPLEX under GAMS terminated at model status {model_status!r} with "
                f"both the iteration limit and the node limit reached; set one of "
                f"the two limits"
            )
        return reached[0]
    check_reported(model_status, (), OUTCOME, "CPLEX under GAMS", "model status")
    return OUTCOME[model_status]


def _option_file(directory: Path, settings: Mapping[str, Any]) -> Path:
    """Write the CPLEX options file of one solve into `directory` and return it."""
    handle, name = tempfile.mkstemp(suffix=".opt", dir=directory)
    with open(handle, "w") as written:
        for key, value in {**FIXED_OPTIONS, **settings}.items():
            written.write(f"{key} {value}\n")
    return Path(name)


def _call_failed(solvelink: int, text: str) -> str:
    """Return the message for a solver call GEV did not make under `solvelink`."""
    check = "run `gams` in a terminal to check the installation and the license"
    if solvelink == CALL_MODULE:
        return f"GEV did not call CPLEX in a separate process: {text}; {check}"
    return (
        f"GEV did not call CPLEX in this process: {text}; pass the option "
        f"separate_process=True, or {check}"
    )


def _remove_new(directory: Path, kept: set[Path]) -> None:
    """Remove every entry of `directory` outside `kept`."""
    for path in set(directory.iterdir()) - kept:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model with CPLEX and return its status, values and bound.

    `Result.backend` is None. `Result.row_dual` is None for a model with
    integer columns. The option `separate_process` runs CPLEX in a process of
    its own. Raises RuntimeError where PATH contains no GAMS system directory,
    where the GAMS license does not run CPLEX, where GEV does not call CPLEX,
    and where CPLEX stops on a failure.
    """
    check_sense(sense)
    settings = translated(options, OPTION_NAMES, OPTION_VALUES, solver="gams")
    call = {name: settings.pop(name, default) for name, default in CALL_OPTIONS.items()}
    given = options or {}
    limits = {
        name: given[name] for name in ("iteration_limit", "node_limit") if name in given
    }
    directory = system_directory()
    library = _library(directory)
    control = _control_file(directory)
    message = ctypes.create_string_buffer(MESSAGE)
    environment, model = _P(), _P()
    library.gevxcreate(ctypes.byref(environment))
    library.gmoxcreate(ctypes.byref(model))
    if not environment.value or not model.value:
        raise RuntimeError(
            f"the GAMS library at {directory} created no GEV or GMO handle; run "
            f"`gams` in a terminal to check the installation"
        )
    scratch = control.parent
    kept = None
    try:
        if library.gevInitEnvironmentLegacy(environment, str(control).encode()) != 0:
            raise RuntimeError(
                f"GEV did not read the control file {control}; run `gams` in a "
                f"terminal to check the license"
            )
        if library.gmoRegisterEnvironment(model, environment, message) != 0:
            raise RuntimeError(
                f"GMO did not register GEV: {_text(message)}; report it as a defect"
            )
        _load(library, model, assembled, sense)
        kept = set(scratch.iterdir())
        option_file = _option_file(scratch, settings)
        library.gmoOptFileSet(model, 1)
        library.gmoNameOptFileSet(model, str(option_file).encode())
        solvelink = int(call["solvelink"])
        logging, log_file = QUIET, None
        if call["logoption"] and solvelink == CALL_MODULE:
            logging, log_file = OWN_FILE, scratch / "solver.log"
        elif call["logoption"]:
            logging = SAME_STREAMS
        failed = library.gevCallSolver(
            environment,
            model,
            b"",
            b"cplex",
            solvelink,
            logging,
            b"" if log_file is None else str(log_file).encode(),
            b"",
            float(call["reslim"]),
            int(call["iterlim"]),
            0,
            float(call["optcr"]),
            float(call["optca"]),
            None,
            message,
        )
        if log_file is not None and log_file.exists():
            sys.stdout.write(log_file.read_text(errors="replace"))
            sys.stdout.flush()
        if failed:
            raise RuntimeError(_call_failed(solvelink, _text(message)))
        return _read(library, model, assembled, limits)
    finally:
        if kept is not None:
            _remove_new(scratch, kept)
        library.gmoxfree(ctypes.byref(model))
        library.gevxfree(ctypes.byref(environment))


def _read(
    library: _Library, model: Any, assembled: "Assembled", limits: Mapping[str, int]
) -> Result:
    """Return the status, the values and the bound GMO reports after a solve.

    Raises RuntimeError where the GAMS license does not run CPLEX.
    """
    code = library.gmoSolveStat(model)
    solve_status = SOLVE_STATUS.get(code, str(code))
    if solve_status == LICENSING:
        raise RuntimeError(
            "GAMS reports licensing problems for CPLEX; run `gams` in a terminal "
            "to read the license"
        )
    code = library.gmoModelStat(model)
    model_status = MODEL_STATUS.get(code, str(code))
    status = _status(
        model_status,
        solve_status,
        limits,
        library.gmoGetHeadnTail(model, ITERATIONS),
        library.gmoGetHeadnTail(model, NODES),
    )
    n, m = assembled.n_cols, assembled.n_rows
    integer = bool(assembled.integrality.any())
    feasible = model_status in FEASIBLE
    col_value = np.zeros(n, dtype=np.float64)
    duals = None
    if CAPABILITIES.reports_duals(integer):
        duals = np.zeros(m, dtype=np.float64)
    objective = 0.0
    if feasible:
        objective = library.gmoGetHeadnTail(model, OBJECTIVE)
        levels = np.empty(n + 1, dtype=np.float64)
        library.gmoGetVarL(model, _address(levels))
        col_value = levels[:n]
        if duals is not None:
            marginals = np.empty(m + 1, dtype=np.float64)
            library.gmoGetEquM(model, _address(marginals))
            duals = marginals[:m]
    bound = None
    if status not in ("infeasible", "unbounded", "unbounded_or_infeasible"):
        reported = library.gmoGetHeadnTail(model, MIP_BOUND) if integer else None
        if reported is not None and abs(reported) >= library.gmoPinf(model):
            reported = np.inf
        bound = proved_bound(status, objective, reported)
    return Result(status, feasible, objective, bound, col_value, duals, None)
