import ast
from pathlib import Path

import nimblend as nb
from docs_blocks import blocks, pages

import nimopt as no

# these adapters read another library's own `index` or `data` -- a scipy
# matrix's and a pandas frame's. Neither library is the package this boundary
# is about, and neither module imports nimblend, so neither can bypass it.
FOREIGN_BUFFERS = ("linopy_models.py", "pypsa_reference.py", "bench_pypsa.py")

# a `Row` has the solver's own row number as `index`, a nimopt field and not
# an array's buffer. The scan reads names, not types, and the modules that
# read one are exempted from it. `test_a_row_reaches_no_array_at_all` covers
# `row.py` instead, the stronger rule: it can read no buffer off an array. The
# others are tests, and each reads a `Row` a function returned.
#
# The exemption covers `index` and `data` alone. `codes` collides with nothing
# nimopt owns, so `test_nimopt_reads_no_domains_codes` scans every source with no
# exemption at all.
OWN_INDEX = (
    "row.py",
    "test_row.py",
    "test_session.py",
    "test_gurobi.py",
    "test_datetime_members.py",
)

# the raw buffers of the layer below the array layer: an array's index matrix
# and value buffer, and a domain's ravelled member codes. Each has a public
# reader above it -- `coordinates()`, `values()`, `positions_of_coordinates()`
# and `as_coord()` -- so reading one bypasses the contract.
BUFFERS = ("index", "data")
CODES = ("codes",)


def modules():
    """Every module of nimopt, as a parsed tree beside its path.

    The whole repository, not the installed package alone: a consumer's tests,
    benchmarks and site tooling reach for nimblend exactly as its source does, and
    a bypass written in a test is a bypass.
    """
    root = Path(no.__file__).parents[2]
    found = []
    for part in ("src", "tests", "benchmarks", "website/scripts"):
        found.extend(sorted((root / part).rglob("*.py")))
    return [(path, ast.parse(path.read_text())) for path in found]


def sources():
    """Return every place this repository writes Python, by its location.

    The modules, and the fenced blocks of the documentation beside them. The
    site documents this boundary on `nimblend/arrays.md` and on
    `for-agents.md`, and `SKILL.md` repeats it for an agent. The examples that
    document the rule are checked against it. A block is identified as
    `<page>:<line>`, where its fence opens.
    """
    found = [(path.name, tree) for path, tree in modules()]
    for page in pages():
        for block in blocks(page.read_text()):
            found.append((f"{page.name}:{block.line}", ast.parse(block.body)))
    return found


def test_the_scan_covers_every_directory_that_holds_python():
    scanned = {path.parts[-2] for path, _ in modules()}
    assert {"nimopt", "tests", "benchmarks", "scripts"} <= scanned


def test_the_scan_covers_every_block_the_suite_executes():
    # the block runner and this scan read the same fences, so an example
    # cannot be executed without also being held to the boundary
    executed = {
        f"{page.name}:{block.line}"
        for page in pages()
        for block in blocks(page.read_text())
    }
    assert executed and executed <= {name for name, _ in sources()}


def test_every_nimblend_import_names_a_public_name_on_the_top_level_module():
    offenders = []
    for name, tree in sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not (module == "nimblend" or module.startswith("nimblend.")):
                    continue
                if module != "nimblend":
                    offenders.append(f"{name}: from {module} import ...")
                    continue
                for alias in node.names:
                    if alias.name not in nb.__all__:
                        offenders.append(f"{name}: {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("nimblend."):
                        offenders.append(f"{name}: import {alias.name}")
    assert offenders == [], offenders


def nimblend_attribute_offenders(tree):
    """Lines reading an attribute of a `nimblend` import alias, off `__all__`.

    `import nimblend` binds the name `nimblend`, and `import nimblend as nb`
    binds `nb`. An attribute of that name, such as `nb.sparse`, reads a
    submodule off the module object; an outer attribute of the result, such
    as `nb.sparse.SparseArray`, is not counted again. A dunder attribute
    such as `nb.__file__` is a module attribute of every module, not a name
    `nimblend.__all__` curates, and is not counted.
    """
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "nimblend"
    }
    return [
        f"{node.lineno} .{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
        and node.attr not in nb.__all__
        and not (node.attr.startswith("__") and node.attr.endswith("__"))
    ]


def test_every_nimblend_attribute_off_an_alias_names_a_public_name():
    offenders = []
    for name, tree in sources():
        offenders.extend(
            f"{name}:{where}" for where in nimblend_attribute_offenders(tree)
        )
    assert offenders == [], offenders


def test_the_attribute_scan_flags_a_submodule_reached_through_an_alias():
    flagged = nimblend_attribute_offenders(
        ast.parse("import nimblend as nb\nx = nb.sparse.SparseArray\n")
    )
    assert flagged == ["2 .sparse"]


def test_the_attribute_scan_accepts_a_public_name_off_the_alias():
    accepted = nimblend_attribute_offenders(
        ast.parse("import nimblend as nb\nnb.SparseArray\n")
    )
    assert accepted == []


def internals_read(tree, names=BUFFERS):
    """Lines reading one of `names` off an object, however it is written.

    A method call of the same name is not a read of a buffer.
    `dims.index(name)` reads a position out of a tuple, and an attribute used
    as a call's function is not counted. Everything else is counted: a
    subscript, an assignment, and an argument passed to another function.
    """
    called = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    return [
        f"{node.lineno} .{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in names
        and id(node) not in called
    ]


def test_nimopt_reads_no_arrays_index_or_data():
    offenders = []
    for name, tree in sources():
        if name in FOREIGN_BUFFERS or name in OWN_INDEX:
            continue
        offenders.extend(f"{name}:{where}" for where in internals_read(tree))
    assert offenders == [], offenders


def test_nimopt_reads_no_domains_codes():
    # a domain's codes are its raw ravelled members, and the site documents
    # the readers above them; no nimopt name is `codes`, and this scan has no
    # exemption
    offenders = []
    for name, tree in sources():
        offenders.extend(f"{name}:{where}" for where in internals_read(tree, CODES))
    assert offenders == [], offenders


# the nimblend names nimopt's own modules need. Reading a buffer is one bypass
# and assembling one is the other: an index matrix built in this package is
# array work done a layer too high, and the adapter interface has to move when
# the kernel below nimblend is replaced. Each entry below is a name that needs
# neither: a domain returns an array over its members, and `from_long` resolves
# label columns through the coordinates a set already has.
PACKAGE_IMPORTS = {
    "DenseArray",
    "Domain",
    "EntryBuffer",
    "ProductCoord",
    "SparseArray",
    "StoredCoord",
    "combined_dims",
    "from_long",
}

# the constructors that take an index matrix and the predicate that checks
# one. They are public, and the site's nimblend reference demonstrates them, so
# this scan is over the package's own modules rather than the repository.
INDEX_CONSTRUCTORS = ("from_canonical", "is_canonical")


# package_modules() covers src/nimopt alone, not tests/ or benchmarks/. A
# benchmark builds a plain numpy array as a fixture before it is passed to
# nimopt, and a test builds one to exercise a case the corpus does not; the
# checks below would flag both as if nimopt had built them.
def package_modules():
    """Every module nimopt ships, as a parsed tree beside its path."""
    root = Path(no.__file__).parent
    return [(path, ast.parse(path.read_text())) for path in sorted(root.rglob("*.py"))]


def test_the_package_imports_only_the_nimblend_names_it_needs():
    imported = {
        alias.name
        for _, tree in package_modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "nimblend"
        for alias in node.names
    }
    assert imported == PACKAGE_IMPORTS, imported


def test_the_package_assembles_no_index_matrix():
    # `SparseArray(index, ...)` and `from_canonical` are how a caller passes
    # nimblend buffers it built itself; a domain returns the array instead
    offenders = []
    for path, tree in package_modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in INDEX_CONSTRUCTORS:
                offenders.append(f"{path.name}:{node.lineno} .{func.attr}")
            elif isinstance(func, ast.Name) and func.id in (
                "SparseArray",
                *INDEX_CONSTRUCTORS,
            ):
                offenders.append(f"{path.name}:{node.lineno} {func.id}")
    assert offenders == [], offenders


def test_the_index_scan_catches_both_spellings():
    caught = []
    tree = ast.parse(
        "SparseArray(idx, v, c, d)\nnimblend.SparseArray.from_canonical(i)\n"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in INDEX_CONSTRUCTORS:
                caught.append(func.attr)
            elif isinstance(func, ast.Name) and func.id == "SparseArray":
                caught.append(func.id)
    assert sorted(caught) == ["SparseArray", "from_canonical"]


def test_a_row_reaches_no_array_at_all():
    # row.py is exempt from the buffer name scan, so it is held to a tighter
    # rule: it reads the assembled vectors and the variables' coordinate
    # rules, and names nothing of nimblend's it could read a buffer from
    source = next(t for p, t in modules() if p.name == "row.py")
    imported = {
        alias.name
        for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "nimblend"
        for alias in node.names
    }
    assert imported == set(), imported


# the attributes that return or call a nimblend domain or coordinate: a
# constraint's `rows`, a variable's `coord` and `domain()`, and their lookups
NIMBLEND_LOOKUPS = {
    "rows",
    "coord",
    "coords",
    "domain",
    "labels",
    "coordinates",
    "to_index",
    "to_position",
    "positions_of",
    "positions_of_coordinates",
}


def test_a_row_calls_no_nimblend_method():
    # row.py reads labels through Constraint and Variable methods
    source = next(t for p, t in modules() if p.name == "row.py")
    reads = [
        f"{node.lineno} .{node.attr}"
        for node in ast.walk(source)
        if isinstance(node, ast.Attribute) and node.attr in NIMBLEND_LOOKUPS
    ]
    assert reads == [], reads


def test_the_rule_catches_a_read_that_is_not_a_subscript():
    # the forms a narrower rule misses: an argument, and a plain assignment
    caught = internals_read(ast.parse("f(arr.index)\nx = arr.data\ny = arr.index[0]\n"))
    assert len(caught) == 3


def test_the_codes_rule_catches_a_read():
    assert internals_read(ast.parse("at = domain.codes\n"), CODES) == ["1 .codes"]


def test_the_rule_passes_a_method_call_of_the_same_name():
    assert internals_read(ast.parse("dims.index('a')\n")) == []


def test_every_model_imports_nimopt_by_its_top_level_module():
    # the models ship in src/ and are read and copied; one that imports past
    # the public surface teaches the habit this file exists to prevent
    offenders = []
    for path, tree in modules():
        if path.parent.name != "models":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            named = node.module or ""
            if named == "nimopt":
                offenders.extend(
                    f"{path.name}: {a.name}"
                    for a in node.names
                    if a.name not in no.__all__
                )
            elif named.startswith("nimopt.") and not named.startswith("nimopt.models"):
                offenders.append(f"{path.name}: from {named} import ...")
    assert offenders == [], offenders


BACKENDS = {"highspy": "highs.py", "gurobipy": "gurobi.py", "mosek": "mosek.py"}


def _imported(tree):
    """Return the top-level name of every import in `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def test_a_backend_is_imported_only_by_the_adapter_that_drives_it():
    # a descriptor is readable whether or not a backend is installed, which
    # holds only while no other module of the package imports one; the
    # adapter named `mosek` shares its backend's name, so what is checked
    # is the import rather than the word
    package = Path(no.__file__).parent
    adapters = {package / "solvers" / module for module in BACKENDS.values()}
    offenders = [
        f"{path.relative_to(package)}: {backend}"
        for path in sorted(package.rglob("*.py"))
        if path not in adapters
        for backend in set(_imported(ast.parse(path.read_text())))
        if backend in BACKENDS
    ]
    assert offenders == [], offenders


def test_each_adapter_names_its_own_backend_and_no_other():
    root = Path(no.__file__).parent / "solvers"
    for backend, module in BACKENDS.items():
        text = (root / module).read_text()
        assert backend in text, module
        for other in set(BACKENDS) - {backend}:
            assert other not in text, (module, other)


def test_a_backend_is_imported_where_it_is_driven_and_not_at_module_scope():
    # `capabilities("gurobi")` returns on a machine with no Gurobi; that is
    # true only while the import is inside the function that needs it
    root = Path(no.__file__).parent / "solvers"
    for backend, module in BACKENDS.items():
        tree = ast.parse((root / module).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert backend not in ast.dump(ast.Module(body=top, type_ignores=[])), module


def test_nothing_a_caller_reads_carries_a_backends_type():
    # the containment: a session holds the solver's model, and everything it
    # returns is nimopt's, numpy's or a builtin
    import numpy as np

    from nimopt import Model, Param, Set, Sum

    T = Set("T", np.arange(2))
    one = Param.from_dense("one", (T,), np.ones(2))
    m = Model("m")
    x = m.var("x", (T,), upper=1.0)
    m.constraint("floor", one[T] * x[T] >= 4.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    with m.session() as session:
        answer = session.solve()
        found = session.diagnose()
    assert found.status == "infeasible"
    allowed = ("builtins", "numpy", "nimblend", "nimopt")
    held = (answer, found, *found.columns, *found.conflict)
    for one_of in held:
        for value in vars(one_of).values():
            assert type(value).__module__.split(".")[0] in allowed, (one_of, value)


# `Domain.from_coordinates` takes an index matrix, exactly as the constructors
# above do. `sets.py` owns the public `subset_of` that wraps it, and every
# other module calls that wrapper instead of building positions of its own.
COORDINATE_CONSTRUCTOR = "from_coordinates"
OWNS_THE_WRAPPER = ("sets.py",)


def test_only_the_module_that_owns_subset_of_builds_a_coordinate_domain():
    offenders = []
    for path, tree in package_modules():
        if path.name in OWNS_THE_WRAPPER:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == COORDINATE_CONSTRUCTOR:
                offenders.append(f"{path.name}:{node.lineno} .{func.attr}")
    assert offenders == [], offenders


# replication of entries across the extent of a dimension is
# `SparseArray.expand`, and an axis reorder is `transpose`. These three numpy
# calls are how the same work is written one layer too high.
REPLICATIONS = ("broadcast_to", "tile", "repeat")


def test_the_package_replicates_entries_through_nimblend():
    offenders = []
    for path, tree in package_modules():
        if path.parent.name == "models":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in REPLICATIONS
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
            ):
                offenders.append(f"{path.name}:{node.lineno} np.{func.attr}")
    assert offenders == [], offenders


def test_a_benchmark_solves_through_a_session():
    # an adapter's solve skips the checks a Session makes: the objective
    # constant and a model that changed after the matrix was assembled
    offenders = []
    for path, tree in modules():
        if path.parent.name not in ("benchmarks", "scripts"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "nimopt.solvers"
            ):
                offenders.append(f"{path.name}: from {node.module} import ...")
            if isinstance(node, ast.ImportFrom) and node.module == "nimopt":
                offenders.extend(
                    f"{path.name}: {a.name}" for a in node.names if a.name == "solvers"
                )
    assert offenders == [], offenders


def test_a_whole_module_import_carries_its_alias():
    aliases = {"nimblend": "nb", "nimopt": "no"}
    offenders = []
    for where, tree in sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import):
                continue
            offenders.extend(
                f"{where}: import {a.name} as {a.asname}"
                for a in node.names
                if a.name in aliases and a.asname != aliases[a.name]
            )
    assert offenders == [], offenders
