"""What nimopt states against what PyPSA states, family by family.

The matrix a builder hands a solver is the thing being compared, so a family
agrees when its rows and its nonzeros both agree. A mismatch is printed beside
the number it should have been rather than left to be hunted.

PyPSA splits its nodal balance into blocks by how many terms a bus carries,
which serves the solver rather than the model, so every block is compared
against the one balance stated here.
"""

import json
from pathlib import Path

from pypsa_network import build, constant

from nimopt.solvers import highs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def group_of(name):
    """The stated family a PyPSA family belongs to."""
    if name.endswith("nodal_balance"):
        return "Bus-nodal_balance"
    return name


def grouped(reference):
    """Each stated family's rows and nonzeros, summed over PyPSA's blocks."""
    totals = {}
    for name, want in reference["families"].items():
        into = group_of(name)
        held = totals.setdefault(into, {"rows": 0, "nnz": 0, "blocks": []})
        held["rows"] += want["rows"]
        held["nnz"] += want["nnz"]
        held["blocks"].append(name)
    return totals


def compare(stem="elec_s_10", root=DATA):
    """Both sides of every family, and the objective each reaches.

    A reference written without a solve carries no objective, and the
    comparison then reports the shape alone.
    """
    root = Path(root)
    reference = json.loads((root / f"{stem}_reference.json").read_text())
    model = build(root / f"{stem}.npz")
    assembled = model.assemble()

    families = {}
    for name, want in grouped(reference).items():
        stated = model.constraints.get(name)
        if stated is None:
            raise ValueError(
                f"PyPSA states {want['blocks']}, which belong to family "
                f"{name!r}; this restatement states no such family"
            )
        families[name] = {
            "rows": (stated.n_rows, want["rows"]),
            "nnz": (stated.nnz, want["nnz"]),
            "blocks": want["blocks"],
        }
    extra = sorted(set(model.constraints) - set(families))
    if extra:
        raise ValueError(f"this restatement states {extra}, which PyPSA does not")

    got = {
        "families": families,
        "rows": (assembled.n_rows, reference["rows"]),
        "cols": (assembled.n_cols, reference["cols"] - 1),
        "nnz": (int(assembled.values.size), reference["nnz"]),
    }
    if "objective" in reference:
        status, value, *_ = highs.solve(assembled, model.sense)
        if status != "optimal":
            raise RuntimeError(f"the restatement solved {status!r}")
        offset = constant(root / f"{stem}.npz")
        got["objective"] = (value - offset, reference["objective"])
    return got


def report(stem="elec_s_10", root=DATA):
    """Print the comparison, marking every line that does not agree."""
    got = compare(stem, root)
    width = max(len(name) for name in got["families"])
    print(f"{'family':{width}} {'rows':>12} {'PyPSA':>9} {'nnz':>10} {'PyPSA':>9}")
    for name, sides in got["families"].items():
        rows, nnz = sides["rows"], sides["nnz"]
        mark = "" if rows[0] == rows[1] and nnz[0] == nnz[1] else "  <-"
        split = f"  ({len(sides['blocks'])} blocks)" if len(sides["blocks"]) > 1 else ""
        print(
            f"{name:{width}} {rows[0]:>12} {rows[1]:>9} "
            f"{nnz[0]:>10} {nnz[1]:>9}{mark}{split}"
        )
    for field in ("rows", "cols", "nnz"):
        mine, theirs = got[field]
        mark = "" if mine == theirs else "  <-"
        print(f"{field:>{width}} {mine:>12} {theirs:>9}{mark}")
    if "objective" in got:
        mine, theirs = got["objective"]
        apart = abs(mine - theirs) / max(1.0, abs(theirs))
        print(f"{'objective':>{width}} {mine:>22,.4f} {theirs:>22,.4f}  ({apart:.2e})")
    return got


if __name__ == "__main__":
    import sys

    stem = sys.argv[1] if len(sys.argv) > 1 else "elec_s_10"
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA
    report(stem, root)
