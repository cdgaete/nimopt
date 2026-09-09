"""The public surface and the refusals, written into the agents page.

The page states the mental model and the mistakes worth prose. What the two
packages export, and every refusal the corpus demonstrates, are read from the
sources instead: a name entering an `__all__`, and a message changing under a
fence, both reach the page by running this.
"""

import re
import sys

import nimblend as nb
from docs_blocks import DOCS, blocks, pages, route

import nimopt as no

PAGE = DOCS / "for-agents.md"

SURFACE = re.compile(r"<!-- surface -->\n.*?<!-- /surface -->", re.S)
REFUSALS = re.compile(r"<!-- refusals -->\n.*?<!-- /refusals -->", re.S)


def catalogue():
    """Every refusal the corpus demonstrates, as type, message and where.

    A refusal shown on more than one page is one entry naming each of them:
    the reader wants the rule once, and the pages that show it beside it.
    """
    found = {}
    for page in pages():
        if page == PAGE:
            continue
        for block in blocks(page.read_text()):
            if block.form != "raises":
                continue
            stated = (block.output or "").strip().splitlines()[-1]
            kind, _, message = stated.partition(": ")
            held = found.setdefault((kind, message), [])
            where = route(page)
            if where not in held:
                held.append(where)
    return sorted(
        ((kind, message, tuple(where)) for (kind, message), where in found.items()),
        key=lambda entry: (entry[2][0], entry[1]),
    )


def exported(package):
    """A package's public names, without the version it also exports."""
    return tuple(name for name in package.__all__ if name != "__version__")


def cell(text):
    """`text` as one cell of a markdown table, its pipes escaped."""
    return text.replace("|", r"\|")


def surface_table():
    """Every public name of both packages, keyed by the module reaching it."""
    rows = ["| From | Names |", "| --- | --- |"]
    for module in (no, nb):
        names = ", ".join(f"`{name}`" for name in exported(module))
        rows.append(f"| `{module.__name__}` | {cell(names)} |")
    return "\n".join(rows)


def refusal_table():
    """Every refusal the corpus demonstrates, as the table the page carries."""
    rows = ["| Raises | Message | Shown at |", "| --- | --- | --- |"]
    for kind, message, where in catalogue():
        shown = " ".join(f"[{at}]({at})" for at in where)
        rows.append(f"| `{kind}` | {cell(message)} | {shown} |")
    return "\n".join(rows)


def rendered(text):
    """`text` with both generated regions written again from their sources.

    A region the page does not open is not added: the markers state where the
    generator's output belongs, and prose owns everything else.
    """
    text = SURFACE.sub(
        lambda _: f"<!-- surface -->\n{surface_table()}\n<!-- /surface -->", text
    )
    return REFUSALS.sub(
        lambda _: f"<!-- refusals -->\n{refusal_table()}\n<!-- /refusals -->", text
    )


def exported_total():
    """Every public name both packages export."""
    return exported(no) + exported(nb)


def main():
    """Rewrite the agents page, reporting what its regions carry."""
    text = PAGE.read_text()
    fresh = rendered(text)
    if fresh != text:
        PAGE.write_text(fresh)
    print(
        f"for-agents.md: {len(exported_total())} public names, "
        f"{len(catalogue())} refusals"
    )


if __name__ == "__main__":
    sys.exit(main())
