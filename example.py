import numpy as np
from tensorhue import viz

from rxtx import (
    pad_matrix,
    split_into_blocks,
    combine_blocks,
    rxtx
)

# Create a random 15x15 matrix
np.random.seed(42)  # for reproducibility
X = np.random.randn(15, 15)
X_padded = pad_matrix(X)
viz(X_padded)

# Matrix with increasing values row-wise
X_structured = np.fromfunction(lambda i, j: i + j, (15, 15), dtype=int)
X_structured_padded = pad_matrix(X_structured)
viz(X_structured_padded)

# Low-rank matrix: rank-3 approximation
A = np.random.randn(15, 3)
X_low_rank = A @ A.T  # Shape will be (15, 15), symmetric and positive semidefinite
X_low_rank_padded = pad_matrix(X_low_rank)
viz(X_low_rank_padded)

# computing results
result_np = X_padded @ X_padded.T
blocks = split_into_blocks(X_padded)
result_rxtx = combine_blocks(rxtx(blocks))
assert np.allclose(result_np, result_rxtx, atol=1e-10), "RXTX differs from np baseline!"

result_np = X_structured_padded @ X_structured_padded.T
blocks = split_into_blocks(X_structured_padded)
result_rxtx = combine_blocks(rxtx(blocks))
assert np.allclose(result_np, result_rxtx, atol=1e-10), "RXTX differs from np baseline!"

result_np = X_low_rank_padded @ X_low_rank_padded.T
blocks = split_into_blocks(X_low_rank_padded)
result_rxtx = combine_blocks(rxtx(blocks))
assert np.allclose(result_np, result_rxtx, atol=1e-10), "RXTX differs from np baseline!"

viz(result_np)
viz(result_rxtx)


diff = result_rxtx - result_np
viz(np.abs(diff))

