"""Dimension names nimopt reserves, and the rules for the names of symbols.

A model's arrays contain the caller's set names and one of these two names.
Both names are double-underscored. No ordinary set name collides with them.
"""

import keyword
from collections.abc import Container, Iterable

COLUMN = "__column__"
ROW = "__row__"


def check_addressable(name: str, what: str) -> None:
    """Raise ValueError for a name an expression cannot refer to.

    An expression refers to a Python identifier that is not a keyword and is
    not Sum.
    """
    if not name.isidentifier() or keyword.iskeyword(name) or name == "Sum":
        raise ValueError(
            f"{what} {name!r} is not a name an expression can address; "
            f"declare a Python identifier other than a keyword or Sum"
        )


def check_one_kind(
    name: str, what: str, declared: Iterable[tuple[str, Container[str]]]
) -> None:
    """Raise ValueError where `name` is declared as another kind of symbol.

    `declared` pairs each other kind with the names declared as that kind.
    """
    for kind, names in declared:
        if name in names:
            raise ValueError(
                f"{what} {name!r} is already declared as a {kind}; declare another name"
            )
