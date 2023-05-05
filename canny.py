from typing import Tuple

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from filters import gaussian_kernel, sobel

import cv2


def _get_idx(quad: int, i: int, j: int, offset: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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
    idx = np.array([i, j]) + np.array(idx) + offset
    return tuple(idx), tuple(-idx)


def non_maxima_suppression(G: np.ndarray, alphas: np.ndarray):
    discrete = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
    get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - discrete)) % 8)
    quadrants = get_quadrant(alphas)


    # padd G with 0, G...G, 0
    G_padded = zero_pad(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            g1_pos, g2_pos = _get_idx(quadrants[i, j], i, j, offset=np.array([1, 1]))
            G[i, j] = G_padded[i+1, j+1] if G_padded[i+1, j+1] >= max(G_padded[g1_pos], G_padded[g2_pos]) else 0

    return G


def zero_pad(arr: np.ndarray) -> np.ndarray:
    padd = np.zeros(np.array(arr.shape) + 2)
    padd[1:-1, 1:-1] = arr.copy()
    return padd


def filter_weak_edges(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    weak = zero_pad(weak)
    strong_indices = list(map(tuple, np.argwhere(strong)))
    while len(strong_indices) > 0:
        i, j = strong_indices.pop(0)
        weak_window = weak[i:i+3, j:j+3]
        weak_edges_relative_to_strong_edge_indices = np.argwhere(weak_window[i:i + 3, j:j + 3]) - 1
        for i_offset, j_offset in weak_edges_relative_to_strong_edge_indices:
            global_idx = i + i_offset, j + j_offset
            strong[global_idx] = 1
            strong_indices.append(global_idx)

    return strong


def save_as_image(img: np.ndarray, location: str) -> None:
    Image.fromarray(img.astype("uint8")).save(location)


def canny(img: np.ndarray, thl: float, thh: float) -> np.ndarray:
    # 1. smooth image
    # maybe don't apply smoothing?
    img = convolve2d(img, gaussian_kernel(5), mode="same", boundary="symm")

    # 2. directional gradients
    G, alphas = sobel(img)

    fig_alphas, ax_alphas = plt.subplots(figsize=(10, 7))
    sns.heatmap(alphas, ax=ax_alphas)
    fig_alphas.savefig('./outputs/alphas.png')

    save_as_image(G, './outputs/gradient_magnitude.png')

    # 3. non maxima suppression
    G = non_maxima_suppression(G, alphas)
    # G  *= (255.0/G.max())
    save_as_image(G, './outputs/non_maxima_suppression.png')

    # 4. double thresholding
    strong = G > thh
    weak = (G > thl) & (~strong)

    edges = strong * 255 + weak * 127
    save_as_image(edges, './outputs/binary_edges.png')

    strong = filter_weak_edges(strong, weak)
    save_as_image(strong*255, "./outputs/canny.png")
    return strong


if __name__ == '__main__':
    # read image as grayscale
    img = np.asarray(Image.open("house1.png").convert("L"))

    edges = canny(img, 10, 69)

    import cv2
    canny = cv2.Canny(img, 10, 69)
    save_as_image(canny, "outputs/groung_tasoihdf.png")


