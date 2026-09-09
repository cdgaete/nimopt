"""A slow, obviously-correct dense form of a model's matrix."""

import numpy as np

from nimopt.names import COLUMN


def dense_matrix(model):
    """The model's constraint matrix built row by row from its constraints."""
    out = np.zeros((model.n_rows, model.n_columns), dtype=np.float64)
    row_start = 0
    for constraint in model.constraints.values():
        block, _ = constraint.expression.materialise()
        at = constraint.rows.positions_of(block)
        columns = block.coordinates()[block.dims.index(COLUMN)]
        values = block.values()
        for k in range(values.size):
            if at[k] >= 0:
                out[row_start + int(at[k]), int(columns[k])] += values[k]
        row_start += constraint.n_rows
    return out
