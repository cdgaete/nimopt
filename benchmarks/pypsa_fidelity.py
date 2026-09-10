"""The nimopt model against the PyPSA model, family by family.

The comparison is over the matrix each builder passes to a solver. A family
agrees when its rows and its nonzeros both agree. A mismatch is printed beside
the expected number.

PyPSA splits its nodal balance into blocks by the number of terms in a bus row.
Every block is compared against the single balance family of the nimopt model.
"""

import json
from pathlib import Path

from pypsa_network import build, constant

from nimopt.solvers import highs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def group_of(name):
    """Return the nimopt family that contains a PyPSA family."""
    if name.endswith("nodal_balance"):
        return "Bus-nodal_balance"
    return name


def grouped(reference):
    """Return the rows and nonzeros of each family, summed over PyPSA blocks."""
    totals = {}
    for name, want in reference["families"].items():
        into = group_of(name)
        group = totals.setdefault(into, {"rows": 0, "nnz": 0, "blocks": []})
        group["rows"] += want["rows"]
        group["nnz"] += want["nnz"]
        group["blocks"].append(name)
    return totals


def compare(stem="elec_s_10", root=DATA):
    """Return both sides of every family and the objective of each side.

    A reference written without a solve has no objective. The comparison then
    reports the shape alone.
    """
    root = Path(root)
    reference = json.loads((root / f"{stem}_reference.json").read_text())
    model = build(root / f"{stem}.npz")
    assembled = model.assemble()

    families = {}
    for name, want in grouped(reference).items():
        constraint = model.constraints.get(name)
        if constraint is None:
            raise ValueError(
                f"family {name!r} of PyPSA blocks {want['blocks']} is absent "
                f"from the nimopt model; declare it in pypsa_network.build"
            )
        families[name] = {
            "rows": (constraint.n_rows, want["rows"]),
            "nnz": (constraint.nnz, want["nnz"]),
            "blocks": want["blocks"],
        }
    extra = sorted(set(model.constraints) - set(families))
    if extra:
        raise ValueError(f"families {extra} are absent from PyPSA; remove them")

    got = {
        "families": families,
        "rows": (assembled.n_rows, reference["rows"]),
        "cols": (assembled.n_cols, reference["cols"] - 1),
        "nnz": (int(assembled.values.size), reference["nnz"]),
    }
    if "objective" in reference:
        result = highs.solve(assembled, model.sense)
        if result.status != "optimal":
            raise RuntimeError(
                f"status is {result.status!r}; compare a network that solves to "
                f"optimality"
            )
        offset = constant(root / f"{stem}.npz")
        got["objective"] = (result.objective - offset, reference["objective"])
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
