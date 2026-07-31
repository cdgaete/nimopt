"""LP/MPS file writers."""

import re

# Characters that are problematic in LP format variable names
_LP_UNSAFE_CHARS = re.compile(r"[-+*/^<>=\s\[\](){}]")


def sanitize_lp_name(name: str) -> str:
    """Sanitize a name for LP format by replacing unsafe characters with underscores."""
    return _LP_UNSAFE_CHARS.sub("_", str(name))


def coord_positions(arr, dim):
    """{element: position} for one dimension of a coefficient array.

    Cached on the array and rebuilt if its coords object is replaced. The
    coefficient lookups run once per term per constraint row, so building and
    scanning a coordinate list there costs O(set size) per call and makes
    export quadratic in the set size.

    Returns None if the coordinates are not hashable, leaving the caller to
    fall back to a linear scan.
    """
    coords = arr.coords[dim]
    cache = arr.__dict__.setdefault("_nimopt_coord_pos", {})
    hit = cache.get(dim)
    if hit is not None and hit[0] is coords:
        return hit[1]

    values = coords.tolist() if hasattr(coords, "tolist") else list(coords)
    try:
        mapping = {}
        for i, v in enumerate(values):
            mapping.setdefault(v, i)  # duplicates: first wins, as list.index does
    except TypeError:
        return None
    cache[dim] = (coords, mapping)
    return mapping
