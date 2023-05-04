import numpy as np


def gaussian_kernel(size: int) -> np.ndarray:
    assert size % 2 != 0, "Size has to be odd"
    g_s = lambda s: np.exp(-s**2/(2*sig**2))
    kernel = np.zeros(size)
    a = (size-1)/2
    sig = a/3
    for i in range(size):
        kernel[i] = g_s(i-a)
    kernel = kernel.reshape(-1, 1) @ kernel.reshape(1, -1)
    return kernel / np.sum(kernel)

gy = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

gx = gy.T