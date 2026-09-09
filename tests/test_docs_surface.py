from pathlib import Path

from docs_blocks import DOCS, route


def test_a_page_is_routed_by_its_path_under_docs():
    assert route(DOCS / "reference" / "param.md") == "/reference/param"


def test_an_index_page_is_routed_by_its_directory():
    assert route(DOCS / "get-started" / "index.md") == "/get-started"


def test_a_stated_slug_wins_over_the_path():
    # the landing page declares `slug: /` and is served there
    assert route(DOCS / "index.md") == "/"


def test_every_page_routes_to_something_absolute():
    from docs_blocks import pages

    assert all(route(p).startswith("/") for p in pages())
    assert isinstance(DOCS, Path)


def test_the_catalogue_names_the_type_the_message_and_where_it_is_shown():
    from surface import catalogue

    found = catalogue()
    assert found, "the corpus demonstrates refusals"
    for kind, message, routes in found:
        assert kind in ("ValueError", "TypeError", "ZeroDivisionError"), kind
        assert message and not message.startswith(kind), message
        assert routes and all(r.startswith("/") for r in routes)


def test_a_refusal_shown_on_two_pages_is_one_entry_naming_both():
    from surface import catalogue

    # the right-hand side frame mismatch is demonstrated in the tutorial and
    # again on the constraint reference page
    entries = [(m, r) for _, m, r in catalogue() if "its right-hand side 'demand'" in m]
    assert len(entries) == 1, entries
    assert set(entries[0][1]) == {"/reference/constraint", "/tutorial/constraints"}


def test_the_catalogue_carries_every_demonstrated_refusal():
    from docs_blocks import blocks, pages
    from surface import catalogue

    shown = {
        (b.output or "").strip().splitlines()[-1]
        for p in pages()
        for b in blocks(p.read_text())
        if b.form == "raises"
    }
    held = {f"{kind}: {message}" for kind, message, _ in catalogue()}
    assert held == shown


def test_the_exported_names_are_the_packages_own_all():
    import nimblend as nb
    from surface import exported

    import nimopt as no

    assert exported(no) == tuple(n for n in no.__all__ if n != "__version__")
    assert "__version__" not in exported(nb)
    assert "SparseArray" in exported(nb)


def test_the_surface_table_states_a_row_per_package():
    from surface import surface_table

    text = surface_table()
    lines = text.splitlines()
    assert lines[0] == "| From | Names |"
    assert lines[1] == "| --- | --- |"
    assert lines[2].startswith("| `nimopt` |")
    assert lines[3].startswith("| `nimblend` |")
    assert "`Definition`" in lines[2]
    assert "`Domain`" in lines[3]
    # a name withdrawn from a package leaves the page by regeneration alone
    assert "`ravel`" not in text


def test_the_refusal_table_states_a_row_per_refusal():
    from surface import catalogue, refusal_table

    text = refusal_table()
    lines = text.splitlines()
    assert lines[0] == "| Raises | Message | Shown at |"
    assert lines[1] == "| --- | --- | --- |"
    assert len(lines) == 2 + len(catalogue())
    assert "`ValueError`" in text


def test_a_message_carrying_a_pipe_cannot_break_the_table():
    from surface import cell

    assert cell("a | b") == r"a \| b"


def test_a_region_is_rewritten_whole():
    from surface import rendered

    stale = (
        "before\n\n"
        "<!-- surface -->\n"
        "| From | Names |\n| --- | --- |\n| `nimopt` | `Gone` |\n"
        "<!-- /surface -->\n\n"
        "after\n"
    )
    fresh = rendered(stale)
    assert "`Gone`" not in fresh
    assert fresh.startswith("before")
    assert fresh.endswith("after\n")
    assert "`Definition`" in fresh


def test_rendering_twice_changes_nothing():
    from surface import PAGE, rendered

    once = rendered(PAGE.read_text())
    assert rendered(once) == once


def test_a_page_without_the_markers_is_untouched():
    from surface import rendered

    plain = "no regions here\n"
    assert rendered(plain) == plain
