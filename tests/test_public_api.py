from pathlib import Path

import nimopt as no


def test_the_module_exports_what_a_model_is_stated_with():
    assert set(no.__all__) == {
        "COLUMN",
        "ROW",
        "Absence",
        "Alias",
        "Assembled",
        "Coefficient",
        "Constraint",
        "Definition",
        "Diagnosis",
        "Explanation",
        "Expression",
        "Model",
        "Option",
        "Param",
        "Relation",
        "Row",
        "Session",
        "Set",
        "Solution",
        "Sum",
        "Term",
        "Variable",
        "__version__",
        "available",
        "capabilities",
        "load",
        "loads",
        "options",
        "product",
        "save",
        "subset",
        "subset_of",
    }


def test_every_exported_name_resolves_on_the_top_level_module():
    absent = [name for name in no.__all__ if not hasattr(no, name)]
    assert absent == [], absent


def test_the_package_ships_its_type_marker():
    assert (Path(no.__file__).parent / "py.typed").is_file()
