"""PDLP on the European network, with a log for a later reader.

`run` builds the nimopt model over the first `--snapshots` snapshots and passes
the matrix to HiGHS with `solver=pdlp`. It writes five files to a directory
named after the run:

- the HiGHS log
- the cuPDLP iteration table and closing report, from the process standard
  output
- one timestamped event per phase
- the GPU memory and utilization, sampled through the solve
- the host resident size, sampled through the solve

`report` reads a run directory and prints:

- the problem presolve reduced, and the device
- the gap and the residuals at checkpoints
- the closing criteria, and the HiGHS check of the point
- the peak GPU memory and the peak host resident size

    python benchmarks/pdlp_gpu.py run --tol 1e-9
    python benchmarks/pdlp_gpu.py report data/large/traces/pdlp/pdlp-all-1e-09

A run without `--time-limit` continues until PDLP stops.
"""

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare import _Watermark, write_trace  # noqa: E402

TRACES = HERE / "data" / "large" / "traces" / "pdlp"

ITERATION = re.compile(
    r"^\s*(?P<iteration>\d+)\s+(?P<primal>[-+.\deE]+)\s+(?P<dual>[-+.\deE]+)\s+"
    r"(?P<gap>[-+.\deE]+)\s+(?P<primal_inf>[-+.\deE]+)\s+(?P<dual_inf>[-+.\deE]+)"
    r"\s+(?P<time>[\d.]+)s?\s+\[(?P<iterate>[AL])\]"
)


def _event(path, **fields):
    """Append one timestamped event to the run's event file."""
    fields = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "clock": time.time(), **fields}
    with path.open("a") as out:
        out.write(json.dumps(fields) + "\n")


def _sample_gpu(path, stop, every=5.0):
    """Write the GPU memory and utilization to `path` every `every` seconds."""
    tool = shutil.which("nvidia-smi")
    with path.open("w") as out:
        out.write("seconds,used_mib,utilization\n")
        started = time.perf_counter()
        while tool and not stop.is_set():
            try:
                read = subprocess.run(
                    [
                        tool,
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                ).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                read = ""
            if read:
                used, utilization = (x.strip() for x in read.split(",")[:2])
                out.write(f"{time.perf_counter() - started:.1f},{used},{utilization}\n")
                out.flush()
            stop.wait(every)


def _info(highs):
    """Return the fields of the HiGHS information record that a report reads.

    A field the record does not define is omitted.
    """
    info = highs.getInfo()
    names = (
        "pdlp_iteration_count",
        "objective_function_value",
        "primal_dual_objective_error",
        "num_primal_infeasibilities",
        "max_primal_infeasibility",
        "sum_primal_infeasibilities",
        "num_dual_infeasibilities",
        "max_dual_infeasibility",
        "sum_dual_infeasibilities",
    )
    return {name: getattr(info, name) for name in names if hasattr(info, name)}


def run(snapshots, tol, time_limit, out):
    """Solve with PDLP and write the log, the events and both memory samples."""
    import highspy
    from bench_pypsa import ARRAYS
    from pypsa_network import Data, model_from

    label = f"pdlp-{snapshots or 'all'}-{tol:g}"
    where = Path(out) / label
    where.mkdir(parents=True, exist_ok=True)
    events = where / "events.jsonl"
    events.write_text("")
    _event(events, event="started", snapshots=snapshots, tol=tol, time_limit=time_limit)

    started = time.perf_counter()
    assembled = model_from(Data(ARRAYS, snapshots)).assemble()
    _event(
        events,
        event="built",
        seconds=round(time.perf_counter() - started, 1),
        rows=int(assembled.n_rows),
        cols=int(assembled.n_cols),
        nnz=int(assembled.values.size),
    )

    lp = highspy.HighsLp()
    lp.num_col_ = assembled.n_cols
    lp.num_row_ = assembled.n_rows
    lp.col_cost_ = assembled.col_cost
    lp.col_lower_ = assembled.col_lower
    lp.col_upper_ = assembled.col_upper
    lp.row_lower_ = assembled.row_lower
    lp.row_upper_ = assembled.row_upper
    matrix = lp.a_matrix_
    matrix.format_ = highspy.MatrixFormat.kRowwise
    matrix.num_col_ = assembled.n_cols
    matrix.num_row_ = assembled.n_rows
    matrix.start_ = assembled.indptr
    matrix.index_ = assembled.indices
    matrix.value_ = assembled.values

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", True)
    highs.setOptionValue("log_to_console", False)
    highs.setOptionValue("log_file", str(where / "highs.log"))
    highs.setOptionValue("solver", "pdlp")
    highs.setOptionValue("pdlp_optimality_tolerance", tol)
    if time_limit:
        highs.setOptionValue("time_limit", float(time_limit))
    highs.passModel(lp)
    del assembled, lp
    _event(
        events, event="passed_to_highs", seconds=round(time.perf_counter() - started, 1)
    )

    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu, args=(where / "gpu.csv", stop), daemon=True
    )
    sampler.start()
    _event(events, event="run_started")
    solve_started = time.perf_counter()
    sys.stdout.flush()
    kept = os.dup(1)
    table = os.open(str(where / "pdlp.out"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(table, 1)
    try:
        with _Watermark(interval=0.05, every=5.0, step=64.0) as mark:
            highs.run()
    finally:
        ctypes.CDLL(None).fflush(None)
        os.dup2(kept, 1)
        os.close(table)
        os.close(kept)
    solve_seconds = time.perf_counter() - solve_started
    stop.set()
    sampler.join()
    write_trace(where / "host.csv", mark.trace)

    status = str(highs.getModelStatus()).split(".")[-1]
    summary = {
        "label": label,
        "snapshots": snapshots,
        "tol": tol,
        "time_limit": time_limit,
        "status": status,
        "solve_seconds": round(solve_seconds, 1),
        "host_rss_peak_mb": round(mark.high),
        "host_rss_added_mb": round(mark.over_baseline),
        **_info(highs),
    }
    _event(events, event="run_finished", **summary)
    (where / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, default=float))
    return summary


def _checkpoints(lines, keep=12):
    """Return at most `keep` iteration rows, evenly spaced, including the last."""
    rows = [m.groupdict() for m in (ITERATION.match(line) for line in lines) if m]
    if len(rows) <= keep:
        return rows
    step = max(1, len(rows) // (keep - 1))
    return rows[::step][: keep - 1] + [rows[-1]]


def report(where):
    """Print the result of a run, read from the files `run` wrote."""
    where = Path(where)

    def lines(name):
        path = where / name
        return path.read_text().splitlines() if path.exists() else []

    log = lines("highs.log")
    table = lines("pdlp.out")
    events = [
        json.loads(line)
        for line in (where / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    print(f"# {where.name}")
    for event in events:
        rest = {k: v for k, v in event.items() if k not in ("at", "clock", "event")}
        shown = json.dumps(rest, default=float)[:200]
        print(f"{event['at']}  {event['event']:16s} {shown}")

    def first(pattern):
        return next((line.strip() for line in log if re.search(pattern, line)), None)

    print("\n## The problem")
    for pattern in (r"^LP has", r"Presolve reductions", r"Solving the presolved"):
        found = first(pattern)
        if found:
            print("  " + found)
    for pattern in (r"Solving with", r"Cuda device", r"Cuda runtime"):
        found = next((line.strip() for line in table if re.search(pattern, line)), None)
        if found:
            print("  " + found)
    presolve = [line.strip() for line in log if re.match(r"^\d+ rows, \d+ cols", line)]
    if presolve:
        print("  presolve passes:\n    " + "\n    ".join(presolve))

    print(
        "\n## Checkpoints: iteration, gap, primal and dual residual, seconds, iterate"
    )
    for row in _checkpoints(table):
        print(
            f"  {int(row['iteration']):>10,}  gap {float(row['gap']):.2e}  "
            f"primal {float(row['primal_inf']):.2e}  "
            f"dual {float(row['dual_inf']):.2e}  "
            f"{float(row['time']):>8.0f}s  [{row['iterate']}]"
        )
    kinds = [m.group("iterate") for m in (ITERATION.match(line) for line in table) if m]
    if kinds:
        average, last = kinds.count("A"), kinds.count("L")
        print(
            f"  rows {len(kinds)}: {average} on the average iterate, {last} on the last"
        )

    print("\n## The PDLP stopping criteria")
    closing = False
    for line in table:
        if line.startswith("Solving information"):
            closing = True
        if closing and line.strip():
            print("  " + line.strip())
        if closing and line.strip().startswith("Number of iterations"):
            break

    print("\n## The HiGHS check of the point")
    for line in log:
        if re.search(
            r"infeasibilit|objective error|Model status changed|WARNING", line
        ):
            print("  " + line.strip())

    for name, unit in (("gpu.csv", "MiB"), ("host.csv", "MB")):
        path = where / name
        if not path.exists():
            continue
        rows = [line.split(",") for line in path.read_text().splitlines()[1:] if line]
        if rows:
            at, peak = max(
                ((float(r[0]), float(r[1])) for r in rows), key=lambda x: x[1]
            )
            print(f"\n## {name}: {len(rows)} samples")
            print(f"  peak {peak:,.0f} {unit} at {at:.0f} s")
    if (where / "summary.json").exists():
        print(
            "\n## Summary\n  "
            + (where / "summary.json").read_text().replace("\n", "\n  ")
        )


def main(argv=None):
    """Parse the command line and dispatch to `run` or `report`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument(
        "--snapshots", type=int, default=None, help="first N; default every one"
    )
    runner.add_argument(
        "--tol", type=float, default=1e-9, help="PDLP's relative tolerance"
    )
    runner.add_argument(
        "--time-limit", type=float, default=None, help="seconds; default none"
    )
    runner.add_argument("--out", type=Path, default=TRACES)
    reader = sub.add_parser("report")
    reader.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.snapshots, args.tol, args.time_limit, args.out)
    else:
        report(args.directory)


if __name__ == "__main__":
    main(sys.argv[1:])
