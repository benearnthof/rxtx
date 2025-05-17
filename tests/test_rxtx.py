import numpy as np
from rxtx import (
    pad_matrix,
    split_into_blocks,
    combine_blocks,
    rxtx
)
def test_rxtx_equivalence():
    X = np.random.randn(16, 16)
    expected = X @ X.T
    blocks = split_into_blocks(X)
    result = combine_blocks(rxtx(blocks))
    assert np.allclose(result, expected, atol=1e-12)
