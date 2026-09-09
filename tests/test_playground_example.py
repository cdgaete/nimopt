"""The code the playground opens with is Python this package runs."""

import re
from pathlib import Path

from docs_blocks import Block
from outputs import shown

import nimopt as no

PLAYGROUND = (
    Path(no.__file__).parents[2] / "website" / "src" / "pages" / "playground.js"
)
START = re.compile(r"const START = `(?P<body>.*?)\n`;", re.DOTALL)


def seed():
    """The example the playground page is served with."""
    found = START.search(PLAYGROUND.read_text())
    assert found, f"{PLAYGROUND.name} states no START template"
    return found.group("body")


def test_the_seed_is_found_at_all():
    assert "from nimopt import" in seed()


def test_the_playground_opens_on_an_example_that_runs():
    # the seed is Python that no page owns, so no docs fence covers it; a
    # reader meets it before anything else and it must not raise
    shown(Block(1, seed(), "run", None, None))
