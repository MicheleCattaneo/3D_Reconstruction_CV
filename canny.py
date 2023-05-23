from typing import Tuple

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from filters import gaussian_kernel, sobel
from utils import zero_pad

import cv2


def _get_idx(quad: int, i: int, j: int, offset: np.ndarray) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Returns the global indices of the neighbours lying in and opposed to the quadrant.
    :param quad: id of quadrant.
    :param i: index of center pixel in x.
    :param j: index of center pixel in y.
    :param offset: numpy array with [x, y] to add onto the global coordinates of the center pixel.
    :return: two index tuples for the two neighbours.
    """
    idx = [
        [0, 1],
        [1, -1],
        [-1, 0],
        [-1, -1],
    ][quad % 4]
    global_idx = np.array([i, j]) + offset
    return tuple(global_idx + idx), tuple(global_idx - idx)


def angular_difference(a: float, b: float) -> float:
    """
    Returns the smaller angular difference in rad between angel a and b.
    Examples:
    >>> angular_difference(0, np.pi)
    π
    >>> angular_difference(-np.pi, np.pi)
    0
    :param a: first angle in rad.
    :param b: second angle in rad.
    :return: minimum angular difference between a and b
    """
    diff_a = np.abs(a - b)
    diff_b = np.pi*2 - diff_a
    return min(diff_a, diff_b)


def non_maximum_suppression(G: np.ndarray, alphas: np.ndarray):
    # define centers of the quadrants
    quadrant_centers = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
    # take absolute difference between the alpha and the quadrant centers
    # get the argmin of the absolute difference to obtain the quadrant id
    get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - quadrant_centers)) % 8)

    # get the quadrant ids
    quadrants = get_quadrant(alphas)

    # padd G with 0, G...G, 0
    G_padded = zero_pad(G)

    # pad alphas for later to ease indexing/ not get out of bound exceptions
    padded_alphas = zero_pad(alphas)
    theta = np.pi / 6  # threshold for angular difference

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            # get indices of the neighbours along the gradient direction and offset by one to account for padding
            g1_pos, g2_pos = _get_idx(quadrants[i, j], i, j, offset=np.array([1, 1]))

            # only consider the neighbour if it belongs to the same edge
            # -> alpha of neighbour and [i, j] is sufficiently close
            G1 = G_padded[g1_pos] if angular_difference(padded_alphas[g1_pos], alphas[i, j]) < theta else 0
            G2 = G_padded[g2_pos] if angular_difference(padded_alphas[g2_pos], alphas[i, j]) < theta else 0

            # apply non-maximum suppression
            G[i, j] = G_padded[i+1, j+1] if G_padded[i+1, j+1] >= max(G1, G2) else 0

    return G


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


def save_as_image(img: np.ndarray, filename: str) -> None:
    """
    Saves image to disk. Domain: [0, 255]
    :param img: image to save.
    :param filename: name of the file.
    """
    if np.max(img) > 255:
        print("\u001b[33mImage has values >255 -> Some bits in the image will overflow!\u001b[0m", "file:", filename)
    Image.fromarray(img.astype("uint8")).save(filename)


def canny(img: np.ndarray, thl: float, thh: float, plot=False) -> np.ndarray:
    # 1. smooth image
    # maybe don't apply smoothing?
    img = convolve2d(img, gaussian_kernel(5), mode="same", boundary="symm")

    # 2. directional gradients
    G, alphas = sobel(img)

    if plot:
        # plot alphas
        fig_alphas, ax_alphas = plt.subplots(figsize=(10, 7))
        sns.heatmap(alphas, ax=ax_alphas)
        fig_alphas.savefig('./outputs/1 alphas.png')

        save_as_image(G, './outputs/2 gradient_magnitude.png')

    # 3. non maxima suppression
    G = non_maximum_suppression(G, alphas)
    if plot:
        save_as_image(G, './outputs/3 non_maxima_suppression.png')

    # 4. double thresholding
    strong = G > thh
    weak = (G > thl) & (~strong)

    if plot:
        # plot strong and weak edges
        edges = strong * 255 + weak * 127
        save_as_image(edges, './outputs/4 strong_weak.png')

    strong = filter_weak_edges(strong, weak)
    if plot:
        save_as_image(strong*255, "./outputs/5 canny_ours.png")

    return strong * 255


if __name__ == '__main__':
    # read image as grayscale
    img = np.asarray(Image.open("house1.png").convert("L"))

    edges = canny(img, 10, 40, plot=True)

    import cv2
    canny = cv2.Canny(img, 10, 40)
    save_as_image(canny, "outputs/6 ground_truth.png")
    save_as_image(((canny-edges)+255) // 2, "outputs/7 difference.png")


