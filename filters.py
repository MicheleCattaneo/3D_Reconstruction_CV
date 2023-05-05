import numpy as np
from scipy.signal import convolve2d

gx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

gy = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])


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


def sobel(img: np.ndarray):
    Gx = convolve2d(img, gx, mode="same", boundary="symm")
    Gy = convolve2d(img, gy, mode="same", boundary="symm")
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G *= 255 / G.max()
    alphas = np.arctan2(Gy, Gx)
    return G, alphas



