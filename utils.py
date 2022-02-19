import numpy as np


def minkowsi_distane(x, y, p=2):
    assert len(x) == len(y), f"Length should be the same. {len(x)=} and {len(y)=}"
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)
