from typing import Tuple

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from filters import gaussian_kernel, gx, gy


def _get_idx(quad: int, i: int, j: int)->Tuple[np.ndarray, np.ndarray]:
    idx = [
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1]
    ][quad]
    idx = np.array(idx) + np.array([i, j])
    return idx, -idx


def non_maxima_suppression(G: np.ndarray, alphas: np.ndarray):
    discrete = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
    get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - discrete)) % 8)
    quadrants = get_quadrant(alphas)

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            g1, g2 = _get_idx(quadrants[i, j], i, j)
            G[i, j] = G[i, j] if G[i, j] >= np.max(G[g1], G[g2]) else 0
    return G


def filter_weak_edges():
    pass


def canny(img: np.ndarray, thl: float, thh: float) -> np.ndarray:
    # 1. smooth image
    # maybe don't apply smoothing?
    img = convolve2d(img, gaussian_kernel(11), mode="same")

    # 2. directional gradients
    Gx = convolve2d(img, gx, mode="same")
    Gy = convolve2d(img, gy, mode="same")
    G = np.sqrt(Gx**2 + Gy**2)
    alphas = np.arctan2(Gy, Gx)

    # 3. non maxima suppression
    G = non_maxima_suppression(G, alphas)

    # 4. double thresholding
    strong = G > thh
    weak = (G > thl) & (~strong)


    print()



if __name__ == '__main__':
    # read image as grayscale
    img = np.asarray(Image.open("house1.png").convert("L"))

    edges = canny(img, .3, .7)

