"""
Basic python implementation of RXTX, the algorithm presented in https://arxiv.org/pdf/2505.09814
"""

import numpy as np

def pad_matrix(mat, block_size=4):
    """
    RXTX uses 4x4 block matrices, we simply pad with 0 until we obtain a matrix that can be split
    evenly into 16 separate blocks.
    """
    n = mat.shape[0]
    pad_to = ((n + block_size - 1) // block_size) * block_size
    pad_width = ((0, pad_to - n), (0, pad_to - n))
    return np.pad(mat, pad_width, mode='constant')

def split_into_blocks(X, blocks_per_row=4):
    """
    Split a padded input matrix into a list of 16 blocks, matching the notation used in the paper.
    """
    n = X.shape[0]
    b = n // blocks_per_row
    blocks = []
    for i in range(blocks_per_row):
        for j in range(blocks_per_row):
            block = X[i*b:(i+1)*b, j*b:(j+1)*b]
            blocks.append(block)
    return blocks  # returns [X1, X2, ..., X16]

def rxtx(blocks):
    """
    Algorithm 1 in https://arxiv.org/pdf/2505.09814
    """
    X1, X2, X3, X4, X5, X6, X7, X8, \
    X9, X10, X11, X12, X13, X14, X15, X16 = blocks
    # Matrix multiplications m1–m26
    m1  = (-X2 + X3 - X4 + X8) @ (X8 + X11).T
    m2  = (X1 - X5 - X6 + X7) @ (X15 + X5).T
    m3  = (-X2 + X12) @ (-X10 + X16 + X12).T
    m4  = (X9 - X6) @ (X13 + X9 - X14).T
    m5  = (X2 + X11) @ (-X6 + X15 - X7).T
    m6  = (X6 + X11) @ (X6 + X7 - X11).T
    m7  = X11 @ (X6 + X7).T
    m8  = X2 @ (-X14 - X10 + X6 - X15 + X7 + X16 + X12).T
    m9  = X6 @ (X13 + X9 - X14 - X10 + X6 + X7 - X11).T
    m10 = (X2 - X3 + X7 + X11 + X4 - X8) @ X11.T
    m11 = (X5 + X6 - X7) @ X5.T
    m12 = (X2 - X3 + X4) @ X8.T
    m13 = (-X1 + X5 + X6 + X3 - X7 + X11) @ X15.T
    m14 = (-X1 + X5 + X6) @ (X13 + X9 + X15).T
    m15 = (X2 + X4 - X8) @ (X11 + X16 + X12).T
    m16 = (X1 - X8) @ (X9 - X16).T
    m17 = X12 @ (X10 - X12).T
    m18 = X9 @ (X13 - X14).T
    m19 = (-X2 + X3) @ (-X15 + X7 + X8).T
    m20 = (X5 + X9 - X8) @ X9.T
    m21 = X8 @ (X9 - X8 + X12).T
    m22 = (-X6 + X7) @ (X5 + X7 - X11).T
    m23 = X1 @ (X13 - X5 + X16).T
    m24 = (-X1 + X4 + X12) @ X16.T
    m25 = (X9 + X2 + X10) @ X14.T
    m26 = (X6 + X10 + X12) @ X10.T

    # Scalar outer products s1–s8
    s1 = X1 @ X1.T
    s2 = X2 @ X2.T
    s3 = X3 @ X3.T
    s4 = X4 @ X4.T
    s5 = X13 @ X13.T
    s6 = X14 @ X14.T
    s7 = X15 @ X15.T
    s8 = X16 @ X16.T

    # Assemble output blocks
    C11 = s1 + s2 + s3 + s4
    C12 = m2 - m5 - m7 + m11 + m12 + m13 + m19
    C13 = m1 + m3 + m12 + m15 + m16 + m17 + m21 - m24
    C14 = m2 - m3 - m5 - m7 - m8 + m11 + m13 - m17 + m23 + m24
    C22 = m1 + m6 - m7 + m10 + m11 + m12 + m22
    C23 = m1 - m4 + m6 - m7 - m9 + m10 + m12 + m18 + m20 + m21
    C24 = m2 + m4 + m11 + m14 + m16 - m18 - m20 + m23
    C33 = m4 - m6 + m7 + m9 - m17 - m18 + m26
    C34 = m3 + m5 + m7 + m8 + m17 + m18 + m25
    C44 = s5 + s6 + s7 + s8

    # Symmetric completion (for readability)
    C21 = C12.T
    C31 = C13.T
    C32 = C23.T
    C41 = C14.T
    C42 = C24.T
    C43 = C34.T

    # Final output: 4x4 block matrix C
    C_blocks = [
        [C11, C12, C13, C14],
        [C21, C22, C23, C24],
        [C31, C32, C33, C34],
        [C41, C42, C43, C44],
    ]

    return C_blocks

def combine_blocks(blocks):
    rows = [np.hstack(row) for row in blocks]
    return np.vstack(rows)
