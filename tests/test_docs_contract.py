import re
from pathlib import Path

import nimblend as nb
from docs_blocks import DOCS, pages, route

import nimopt as no

STATIC = Path(__file__).parents[1] / "website" / "static"


def test_the_index_names_every_page():
    index = (STATIC / "llms.txt").read_text()
    missing = [route(p) for p in pages() if f"]({route(p)})" not in index]
    assert missing == [], missing


def test_the_full_text_carries_every_page():
    full = (STATIC / "llms-full.txt").read_text()
    missing = [route(p) for p in pages() if f"# {route(p)}\n" not in full]
    assert missing == [], missing


SKILL = Path(__file__).parents[1] / ".claude" / "skills" / "nimopt" / "SKILL.md"


def body(text):
    """The page below its frontmatter."""
    if not text.startswith("---\n"):
        return text.strip()
    return text.split("\n---\n", 1)[1].strip()


def test_the_skill_is_the_agents_page():
    page = (DOCS / "for-agents.md").read_text()
    assert body(SKILL.read_text()) == body(page)


def test_the_skill_declares_a_name_and_a_description():
    head = SKILL.read_text().split("\n---\n", 1)[0]
    assert "name: nimopt" in head
    assert "description:" in head


REFERENCE = DOCS / "reference"
NOT_DOCUMENTED = {"__version__"}
HOME = {
    "Absence": "inspection",
    "Model": "model",
    "Assembled": "model",
    "Set": "sets",
    "Alias": "sets",
    "product": "sets",
    "subset": "sets",
    "subset_of": "sets",
    "Param": "param",
    "Coefficient": "param",
    "Variable": "variable",
    "COLUMN": "variable",
    "ROW": "variable",
    "Explanation": "explanation",
    "Expression": "expression",
    "Term": "expression",
    "Sum": "expression",
    "Relation": "expression",
    "Row": "inspection",
    "Diagnosis": "solvers",
    "Option": "solvers",
    "options": "solvers",
    "Session": "solvers",
    "available": "solvers",
    "capabilities": "solvers",
    "Constraint": "constraint",
    "Definition": "definition",
    "Solution": "solution",
    "load": "files",
    "loads": "files",
    "save": "files",
}


def test_every_public_nimopt_name_has_a_home_page():
    assert set(no.__all__) - NOT_DOCUMENTED == set(HOME)


def test_each_name_is_documented_on_its_own_page():
    missing = [
        name
        for name, page in HOME.items()
        if f"`{name}`" not in (REFERENCE / f"{page}.md").read_text()
    ]
    assert missing == [], missing


NIMBLEND = DOCS / "nimblend"
NIMBLEND_HOME = {
    "Array": "arrays",
    "SparseArray": "arrays",
    "DenseArray": "arrays",
    "from_long": "arrays",
    "from_dense": "arrays",
    "is_canonical": "arrays",
    "combined_dims": "arrays",
    "Domain": "domains",
    "StoredCoord": "domains",
    "ProductCoord": "domains",
    "SubsetCoord": "domains",
    "EntryBuffer": "domains",
}


def test_every_public_nimblend_name_has_a_home_page():
    # the rule nimopt's own names live under, over the package below it too
    assert set(nb.__all__) - NOT_DOCUMENTED == set(NIMBLEND_HOME)


def test_each_nimblend_name_is_documented_on_its_own_page():
    missing = [
        name
        for name, page in NIMBLEND_HOME.items()
        if f"`{name}`" not in (NIMBLEND / f"{page}.md").read_text()
    ]
    assert missing == [], missing


README = Path(__file__).parents[1] / "README.md"
SITE_ONLY = {"/playground"}
ROUTES = {
    "/get-started": "website/docs/get-started/index.md",
    "/vocabulary": "website/docs/vocabulary/index.md",
    "/nimblend": "website/docs/nimblend/index.md",
    "/tutorial/sets-and-parameters": "website/docs/tutorial/sets-and-parameters.md",
    "/explanation/what-the-numbers-measure": (
        "website/docs/explanation/what-the-numbers-measure.md"
    ),
    "/for-agents": "website/docs/for-agents.md",
}
LINK = re.compile(r"\[([^\]]*)\]\((/[^)]*)\)")


def as_readme(page):
    """The README the front page states: its body, with the site's routes
    answered by the files a reader on the repository has."""

    def repointed(match):
        label, route = match.group(1), match.group(2)
        if route in SITE_ONLY:
            return label
        return f"[{label}]({ROUTES[route]})"

    return LINK.sub(repointed, body(page)) + "\n"


def test_the_readme_is_the_front_page():
    assert README.read_text() == as_readme((DOCS / "index.md").read_text())


def test_every_route_the_front_page_states_is_answered():
    stated = {route for _, route in LINK.findall((DOCS / "index.md").read_text())}
    assert stated <= (set(ROUTES) | SITE_ONLY), stated - set(ROUTES) - SITE_ONLY


def test_the_front_page_is_the_site_root():
    assert "slug: /" in (DOCS / "index.md").read_text()


def test_the_agents_page_carries_the_regions_a_fresh_run_writes():
    # the same rule the output regions live under: a page is a function of
    # its prose and its sources, and a stale region fails rather than lies
    from surface import PAGE, rendered

    text = PAGE.read_text()
    assert rendered(text) == text


def test_every_refusal_the_corpus_demonstrates_reaches_the_agents_page():
    from surface import PAGE, catalogue

    page = PAGE.read_text()
    missing = [message for _, message, _ in catalogue() if message not in page]
    assert missing == [], missing


def test_the_agents_page_states_every_public_name_of_both_packages():
    from surface import PAGE, exported

    page = PAGE.read_text()
    missing = [
        name
        for module in (no, nb)
        for name in exported(module)
        if f"`{name}`" not in page
    ]
    assert missing == [], missing


def test_the_index_leads_with_the_page_an_agent_reads_first():
    # llms.txt is the agent's index, so the surface leads it
    index = (STATIC / "llms.txt").read_text()
    sections = [line for line in index.splitlines() if line.startswith("## ")]
    assert sections[0] == "## For agents", sections


def test_the_sidebar_teaches_before_it_explains():
    # the sidebar serves a reader learning to build a model: the tutorial and
    # the guides come before the pages that argue why the design is sound
    sidebar = (Path(__file__).parents[1] / "website" / "sidebars.ts").read_text()
    at_tutorial = sidebar.index('label: "Tutorial"')
    at_guides = sidebar.index('label: "Guides"')
    at_explanation = sidebar.index('label: "Explanation"')
    assert at_tutorial < at_guides < at_explanation, "Tutorial, Guides, Explanation"


def test_the_landing_page_names_the_capabilities_a_reader_chooses_on():
    # the front page argues what is different; the verbs it argues from are
    # the ones a reader would otherwise not know exist
    landing = (DOCS / "index.md").read_text()
    missing = [
        name
        for name in ("Definition", "Session", "diagnose", "absent", "explain", "Gurobi")
        if name not in landing
    ]
    assert missing == [], missing


def test_the_solvers_page_carries_the_option_region_a_fresh_run_writes():
    from solver_options import PAGE, rendered

    text = PAGE.read_text()
    assert rendered(text) == text


def test_every_option_reaches_the_solvers_page():
    from nimopt import options

    page = (REFERENCE / "solvers.md").read_text()
    missing = [o.name for o in options() if f"`{o.name}`" not in page]
    assert missing == [], missing
