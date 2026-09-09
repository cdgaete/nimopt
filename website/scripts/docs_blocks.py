"""The Python blocks a documentation page carries, and the form each states."""

import re
from dataclasses import dataclass
from pathlib import Path

FENCE = re.compile(r"^```python(?P<meta>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.S | re.M)
RAISES = re.compile(r"^raises=(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
SKIP = re.compile(r'^skip="(?P<reason>[^"]+)"$')
SLUG = re.compile(r"^slug:\s*(\S+)\s*$", re.M)

REGION = re.compile(
    r"\n\n<!-- output -->\n<details open>\n<summary>(?P<summary>[^\n<]*)</summary>\n\n"
    r"```text\n(?P<text>.*?)\n```\n\n</details>\n<!-- /output -->",
    re.S,
)

DOCS = Path(__file__).parents[1] / "docs"


@dataclass(frozen=True)
class Block:
    """One fenced block: where it starts, what it holds, how it is run, what it shows.

    `output` is the text of the generated region below the fence -- what the
    block printed and the error it raised -- or None where the page carries
    no region for it.
    """

    line: int
    body: str
    form: str
    detail: str | None
    output: str | None


def pages():
    """Every documentation page, in a stable order."""
    return sorted(DOCS.rglob("*.md"))


def route(page):
    """The site route a page is served at.

    A page states its own route with `slug:` in its frontmatter; otherwise it
    is served at its path under `docs/`, without the suffix and without a
    trailing `index`.
    """
    text = page.read_text()
    if text.startswith("---\n"):
        stated = SLUG.search(text.split("\n---\n", 1)[0])
        if stated:
            return stated.group(1)
    parts = page.relative_to(DOCS).with_suffix("")
    return "/" + str(parts).removesuffix("/index")


def blocks(text):
    """The Python blocks in `text`, refusing a fence whose form is unknown."""
    found = []
    for match in FENCE.finditer(text):
        meta = match.group("meta").strip()
        line = text.count("\n", 0, match.start()) + 1
        if not meta:
            form, detail = "run", None
        elif RAISES.match(meta):
            form, detail = "raises", RAISES.match(meta).group("name")
        elif SKIP.match(meta):
            form, detail = "skip", SKIP.match(meta).group("reason")
        else:
            raise ValueError(
                f"line {line}: the fence states {meta!r}; a python block is "
                f'plain, `raises=<Error>`, or `skip="<reason>"`'
            )
        region = REGION.match(text, match.end())
        output = region.group("text") if region else None
        found.append(Block(line, match.group("body"), form, detail, output))
    return found


def strip_regions(text):
    """`text` with every generated output region removed."""
    return REGION.sub("", text)
