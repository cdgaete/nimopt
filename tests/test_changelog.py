import re
from datetime import date
from pathlib import Path

import nimopt as no

CHANGELOG = Path(__file__).parents[1] / "CHANGELOG.md"
HEADING = re.compile(r"^## (?:(Unreleased)|(\d+\.\d+\.\d+) - (\d{4}-\d{2}-\d{2}))$")


def sections():
    """Each `##` section as (version, date, body); `Unreleased` has no date."""
    found = []
    for line in CHANGELOG.read_text().splitlines():
        match = HEADING.match(line)
        if match:
            unreleased, version, day = match.groups()
            found.append([unreleased or version, day, []])
        elif line.startswith("## "):
            raise AssertionError(f"a heading the format does not define: {line!r}")
        elif found:
            found[-1][2].append(line)
    return [(v, d, "\n".join(b).strip()) for v, d, b in found]


def released():
    return [s for s in sections() if s[0] != "Unreleased"]


def test_the_unreleased_section_comes_first():
    assert sections()[0][0] == "Unreleased"


def test_the_latest_release_is_the_version_the_package_states():
    # a version bump without a changelog section, or a section without the
    # bump, fails here
    assert released()[0][0] == no.__version__


def test_every_release_carries_a_date_and_at_least_one_entry():
    for version, day, body in released():
        date.fromisoformat(day)
        assert "\n- " in "\n" + body, version


def test_releases_descend():
    versions = [tuple(int(n) for n in v.split(".")) for v, _, _ in released()]
    assert versions == sorted(versions, reverse=True)
    assert len(versions) == len(set(versions))
