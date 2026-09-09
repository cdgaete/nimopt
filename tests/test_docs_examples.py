import textwrap

import pytest
from docs_blocks import DOCS, Block, blocks, pages
from outputs import rendered, shown


def test_a_plain_block_is_run():
    found = blocks(
        textwrap.dedent("""\
        text
        ```python
        x = 1
        ```
        """)
    )
    assert [(b.form, b.body.strip()) for b in found] == [("run", "x = 1")]


def test_a_block_can_state_the_error_it_raises():
    found = blocks(
        textwrap.dedent("""\
        ```python raises=ValueError
        raise ValueError("no")
        ```
        """)
    )
    assert found[0].form == "raises"
    assert found[0].detail == "ValueError"


def test_a_skipped_block_states_a_reason():
    found = blocks(
        textwrap.dedent("""\
        ```python skip="needs a solver licence"
        solve()
        ```
        """)
    )
    assert found[0].form == "skip"
    assert found[0].detail == "needs a solver licence"


def test_an_unknown_form_is_refused():
    with pytest.raises(ValueError, match="fence"):
        blocks("```python maybe=yes\nx = 1\n```\n")


def test_a_block_carries_the_line_it_starts_on():
    found = blocks("intro\n\n```python\nx = 1\n```\n")
    assert found[0].line == 3


def test_only_python_fences_are_collected():
    assert blocks("```bash\nls\n```\n") == []


def cases():
    """Every runnable block on every page, identified by page and line."""
    found = []
    for page in pages():
        for block in blocks(page.read_text()):
            if block.form != "skip":
                name = f"{page.relative_to(DOCS)}:{block.line}"
                found.append(pytest.param(block, id=name))
    return found


@pytest.mark.parametrize("block", cases())
def test_every_example_behaves_as_the_page_states(block):
    # shown() runs the block and refuses one that defies its fence, so the
    # form and the text the page shows are asserted in the same pass
    shows = shown(block)
    assert block.output == (shows[1] if shows is not None else None)


def test_every_page_carries_the_regions_a_fresh_run_writes():
    # catches a region no fence owns, which the per-block cases never visit
    stale = [p.name for p in pages() if rendered(p.read_text()) != p.read_text()]
    assert stale == [], stale


def test_the_pages_are_found_at_all():
    assert pages(), f"no pages under {DOCS}"


def test_a_fence_naming_the_wrong_error_is_refused():
    # the name on the fence is matched against what the block actually raises,
    # so a typo in it fails that block rather than passing unnoticed
    wrong = Block(1, 'raise ValueError("no")', "raises", "ValuError", None)
    with pytest.raises(ValueError, match="states raises=ValuError"):
        shown(wrong)
