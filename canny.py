from typing import Tuple

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from filters import gaussian_kernel, gx, gy


def _get_idx(quad: int, i: int, j: int, offset: np.ndarray)->Tuple[Tuple[int, int], Tuple[int, int]]:
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
    idx = np.array(idx) + np.array([i, j]) + offset
    return tuple(idx), tuple(-idx)

# def in_range(position, max_y, max_x):
#     return 0 <= position[0] < max_y and 0 <= position[1] < max_x

# def get_offsetted_indices(i,j, offset_xy, max_y, max_x)->Tuple or None:

#     new_position = np.array([i,j]) + offset_xy

#     return (new_position[0], new_position[1]) if in_range(new_position, max_y, max_x) else None


def non_maxima_suppression(G: np.ndarray, alphas: np.ndarray):
    discrete = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
    get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - discrete)) % 8)
    quadrants = get_quadrant(alphas)

    print(G.shape)
    # padd G with 0, G...G, 0
    G_padded = np.zeros(np.array(G.shape)+2)
    G_padded[1:-1, 1:-1] = G
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            g1_pos, g2_pos = _get_idx(quadrants[i, j], i, j, offset=np.array([1, 1]))
            # g1_pos = get_offsetted_indices(i,j, dir1, G.shape[0], G.shape[1])
            # g2_pos = get_offsetted_indices(i,j, dir2, G.shape[0], G.shape[1])

            # if g1_pos is not None and g2_pos is not None:
            G[i, j] = G[i, j] if G[i, j] >= max(G_padded[g1_pos], G_padded[g2_pos]) else 0

    return G


def filter_weak_edges():
    pass

def save_as_image(img: np.ndarray, location: str) -> None:
    Image.fromarray(img.astype("uint8")).save(location)

def canny(img: np.ndarray, thl: float, thh: float) -> np.ndarray:
    # 1. smooth image
    # maybe don't apply smoothing?
    img = convolve2d(img, gaussian_kernel(11), mode="same")

    # 2. directional gradients
    Gx = convolve2d(img, gx, mode="same")
    Gy = convolve2d(img, gy, mode="same")
    G = np.sqrt(Gx**2 + Gy**2)
    alphas = np.arctan2(Gy, Gx)
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

    print(G.min(), G.max())



if __name__ == '__main__':
    # read image as grayscale
    img = np.asarray(Image.open("house1.png").convert("L"))

    edges = canny(img, 100, 200)

