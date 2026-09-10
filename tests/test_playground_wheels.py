"""The wheels the playground installs are the packages this repo builds."""

import re
import zipfile
from pathlib import Path

import nimblend as nb

import nimopt as no

WHEELS = Path(no.__file__).parents[2] / "website" / "static" / "wheels"
NAME = re.compile(r"^(?P<package>[a-z]+)-(?P<version>[^-]+)-py3-none-any\.whl$")
SHIPPED = {"nimblend": nb, "nimopt": no}


def wheels():
    """Every wheel the site serves, by package name."""
    found = {}
    for path in sorted(WHEELS.glob("*.whl")):
        match = NAME.match(path.name)
        assert match, f"{path.name} is not a universal wheel this test can read"
        found[match.group("package")] = match.group("version")
    return found


def test_the_site_ships_a_wheel_for_each_package_the_playground_imports():
    assert set(wheels()) == set(SHIPPED)


def test_each_wheel_carries_the_version_its_package_states():
    # a wheel is a build artefact of src/, and nothing rebuilds it on import;
    # this fails the release that bumps a version without rebuilding
    stale = {
        package: (version, SHIPPED[package].__version__)
        for package, version in wheels().items()
        if version != SHIPPED[package].__version__
    }
    assert stale == {}, stale


def test_every_wheel_is_universal_so_a_browser_can_install_it():
    # nimblend and nimopt are pure Python; a wheel built for a platform would
    # install here and fail in the browser, where the platform is wasm32
    assert all(path.name.endswith("-py3-none-any.whl") for path in WHEELS.glob("*.whl"))


def modules(package):
    """Every module a package ships, by its path inside a wheel."""
    root = Path(SHIPPED[package].__file__).parent
    return {
        f"{package}/{path.relative_to(root)}": path.read_bytes()
        for path in root.rglob("*.py")
    }


def packed(package):
    """Return every module in the shipped wheel, by its path inside it."""
    whl = WHEELS / f"{package}-{SHIPPED[package].__version__}-py3-none-any.whl"
    with zipfile.ZipFile(whl) as held:
        return {n: held.read(n) for n in held.namelist() if n.endswith(".py")}


def test_each_wheel_carries_the_modules_its_package_states():
    # a version comparison passes a wheel built before a change that kept the
    # version, and the playground then runs code this repository does not have
    missing = {p: sorted(set(modules(p)) - set(packed(p))) for p in SHIPPED}
    assert missing == {p: [] for p in SHIPPED}, missing


def test_each_wheel_carries_what_its_package_says_those_modules_are():
    stale = {
        package: sorted(
            name
            for name, body in modules(package).items()
            if packed(package).get(name) != body
        )
        for package in SHIPPED
    }
    assert stale == {p: [] for p in SHIPPED}, stale
