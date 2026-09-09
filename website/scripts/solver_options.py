"""The option table, written into the solvers reference page.

The page states what an option means; which name each solver gives it is read
from the adapters, so an option entering the vocabulary reaches the page by
running this.
"""

import re
import sys

from docs_blocks import DOCS

import nimopt as no

PAGE = DOCS / "reference" / "solvers.md"

REGION = re.compile(r"<!-- options -->\n.*?<!-- /options -->", re.S)

SOLVERS = ("highs", "gurobi", "mosek")


def cell(text):
    """`text` as one cell of a markdown table, its pipes escaped."""
    return str(text).replace("|", r"\|")


def table():
    """Every option, what it takes, and the name each solver gives it."""
    rows = [
        "| Option | Takes | Does | " + " | ".join(f"`{s}`" for s in SOLVERS) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in SOLVERS) + " |",
    ]
    native = {s: {o.name: o.native for o in no.options(s)} for s in SOLVERS}
    for option in no.options():
        takes = (
            " / ".join(f"`{c}`" for c in option.choices)
            if option.choices
            else option.kind
        )
        spellings = " | ".join(
            f"`{native[s][option.name]}`"
            if native[s][option.name] is not None
            else "not carried"
            for s in SOLVERS
        )
        rows.append(
            f"| `{option.name}` | {takes} | {cell(option.does)} | {spellings} |"
        )
    return "\n".join(rows)


def rendered(text):
    """`text` with the option region written again from the vocabulary."""
    return REGION.sub(lambda _: f"<!-- options -->\n{table()}\n<!-- /options -->", text)


def main():
    """Rewrite the solvers page, reporting what its region carries."""
    text = PAGE.read_text()
    fresh = rendered(text)
    if fresh != text:
        PAGE.write_text(fresh)
    print(f"solvers.md: {len(no.options())} options")


if __name__ == "__main__":
    sys.exit(main())
