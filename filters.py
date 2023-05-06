import numpy as np
from scipy.signal import convolve2d

g3x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

g3y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

g5y = np.array([
    [2, 2, 4, 2, 2],
    [1, 1, 2, 1, 1],
    [0, 0, 0, 0, 0],
    [-1, -1, -2, -1, -1],
    [-2, -2, -4, -2, -2]
])

g5x = g5y.T


def gaussian_kernel(size: int) -> np.ndarray:
    assert size % 2 != 0, "Size has to be odd"
    g_s = lambda s: np.exp(-s ** 2 / (2 * sig ** 2))
    kernel = np.zeros(size)
    a = (size - 1) / 2
    sig = a / 3
    for i in range(size):
        kernel[i] = g_s(i - a)
    kernel = kernel.reshape(-1, 1) @ kernel.reshape(1, -1)
    return kernel / np.sum(kernel)


def sobel(img: np.ndarray, size: str = "3x3"):
    gx, gy = (g3x, g3y) if size == "3x3" else (g5x, g5y)

    Gx = convolve2d(img, gx, mode="same", boundary="symm")
    Gy = convolve2d(img, gy, mode="same", boundary="symm")
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G *= 255 / G.max()
    alphas = np.arctan2(Gy, Gx)
    return G, alphas
