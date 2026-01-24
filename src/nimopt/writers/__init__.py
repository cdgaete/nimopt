"""LP/MPS file writers."""

import re

# Characters that are problematic in LP format variable names
_LP_UNSAFE_CHARS = re.compile(r"[-+*/^<>=\s\[\](){}]")


def sanitize_lp_name(name: str) -> str:
    """Sanitize a name for LP format by replacing unsafe characters with underscores."""
    return _LP_UNSAFE_CHARS.sub("_", str(name))
