from typing import Tuple

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from filters import gaussian_kernel, sobel

import cv2


def _get_idx(quad: int, i: int, j: int, offset: np.ndarray) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    idx = [
        [0, 1],
        [1, -1],
        [-1, 0],
        [-1, -1],
    ][quad % 4]
    global_idx = np.array([i, j]) + offset
    return tuple(global_idx + idx), tuple(global_idx - idx)


def non_maxima_suppression2(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def angular_difference(a: float, b: float) -> float:
    diff_a = np.abs(a - b)
    diff_b = np.pi*2 - diff_a
    return min(diff_a, diff_b)


def non_maxima_suppression(G: np.ndarray, alphas: np.ndarray):
    discrete = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
    get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - discrete)) % 8)

    quadrants = get_quadrant(alphas) % 4
    padded_alphas = zero_pad(alphas)

    # padd G with 0, G...G, 0
    G_padded = zero_pad(G)

    # G_new = G.copy()
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            g1_pos, g2_pos = _get_idx(quadrants[i, j], i, j, offset=np.array([1, 1]))
            theta = np.pi / 6
            G1 = G_padded[g1_pos] if angular_difference(padded_alphas[g1_pos], alphas[i, j]) < theta else 0
            G2 = G_padded[g2_pos] if angular_difference(padded_alphas[g2_pos], alphas[i, j]) < theta else 0

            G[i, j] = G_padded[i+1, j+1] if G_padded[i+1, j+1] >= max(G1, G2) else 0

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
        # get 3x3 window around the strong index in the weak matrix as strong candidates
        weak_window = weak[i:i+3, j:j+3]
        # extract the indices of the weak edges in the window
        weak_edges_relative_to_strong_edge_indices = np.argwhere(weak_window) - 1
        # set the weak edges as strong edges in the strong matrix and add their indices to the queue
        for i_offset, j_offset in weak_edges_relative_to_strong_edge_indices:
            global_idx = i + i_offset, j + j_offset
            strong[global_idx] = 1
            strong_indices.append(global_idx)

        # remove processed weak edges
        weak[i:i + 3, j:j + 3] = 0

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
    fig_alphas.savefig('./outputs/1 alphas.png')

    save_as_image(G, './outputs/2 gradient_magnitude.png')

    # 3. non maxima suppression
    G = non_maxima_suppression(G, alphas)
    # G  *= (255.0/G.max())
    save_as_image(G, './outputs/3 non_maxima_suppression.png')

    # 4. double thresholding
    strong = G > thh
    weak = (G > thl) & (~strong)

    edges = strong * 255 + weak * 127
    save_as_image(edges, './outputs/4 strong_weak.png')

    strong = filter_weak_edges(strong, weak)
    save_as_image(strong*255, "./outputs/5 canny_ours.png")
    return strong * 255


if __name__ == '__main__':
    # read image as grayscale
    img = np.asarray(Image.open("house1.png").convert("L"))

    edges = canny(img, 10, 69)

    import cv2
    canny = cv2.Canny(img, 10, 69)
    save_as_image(canny, "outputs/6 groung_tasoihdf.png")
    save_as_image(((canny-edges)+255) // 2, "outputs/7 difference.png")


