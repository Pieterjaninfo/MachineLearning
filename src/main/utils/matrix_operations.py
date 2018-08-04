import numpy as np


def matrix_equal(a, b):
    if a.shape == b.shape:
        eq = np.equal(a, b)
        return eq[np.where(eq == False)].size == 0
    return False
