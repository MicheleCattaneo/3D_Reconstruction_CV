from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import zero_pad, ishow


def filter_edges_by_color(image: np.ndarray, edges: np.ndarray, colors: List[np.ndarray]):
    """
    Retains edge iff at least two colors of the colors list are in the 15x15 pixel neighbourhood of the edge.
    :param image: colored image
    :param edges: corresponding edges to the image
    :param colors: colors that have to be in the neighbourhood of the wanted edges
    :return: mask of final edges
    """
    mask = np.zeros_like(edges, dtype=bool)
    padded_img = zero_pad(image, 3)
    dist = lambda M, x: np.sum(np.abs(M - x), axis=2)

    for i, j in np.argwhere(edges):
        dists_to_colors = list(map(lambda c: np.min(dist(padded_img[i:i + 7, j:j + 7], c)), colors))

        mask[i, j] = np.sum(np.array(dists_to_colors) < 12) > 1
    return mask


def get_maxima(hough: np.ndarray, threshold: int, window: int = 10) -> np.ndarray:
    """
    Apply non maxima suppression in the hough space to eliminate lines with similar angles.
    :param hough: hough space to go through optimas
    :param threshold: threshold preselect candidates
    :param window: window to consider for the suppression
    :return: line definitions with nx2 with (theta, rho)
    """
    mask = hough > threshold
    padded = zero_pad(hough, window)

    for i, j in np.argwhere(mask):
        mask[i, j] = hough[i, j] >= padded[i:i + 2 * window + 1, j:j + 2 * window + 1].max()

    return np.argwhere(mask)


def guided_hough(
        edges: np.ndarray, guide: np.ndarray,
        voting_threshold: int
) -> np.ndarray:
    """
    Implements a hough transform based on predefined color filtering.
    :param edges: extracted edges
    :param guide: colored image which is used to guide the filtering of the edges
    :param voting_threshold: number of votes needed to be considered a line
    :return:  definitions as an array of nx2 with (theta, rho)
    """
    mask = filter_edges_by_color(guide, edges, [
        np.array([0, 0, 224]),
        np.array([235, 237, 238]),
        np.array([85, 146, 197]),
        np.array([211, 0, 0]),
        np.array([43, 108, 172])
    ])

    window_size = 10

    # plot mask
    plt.imshow(mask, cmap="gray")
    plt.title("Selected Edges")
    plt.savefig("outputs/filtered_edges.png")
    plt.show()

    # create empty hough space with a padded window in the height which is needed for a reliable NMS
    hypot = np.ceil(np.hypot(*mask.shape))
    H = np.zeros((180+2*window_size, int(hypot*2)))

    for i, j in np.argwhere(mask):
        for theta in range(-window_size, 180+window_size):
            theta_rad = theta * np.pi/180
            p = np.cos(theta_rad) * j + np.sin(theta_rad) * i
            H[theta+window_size, int(np.round(p) + hypot)] += 1

    # select lines by thresholding and eliminate non-local maxima
    our_lines = get_maxima(H, voting_threshold, window_size)
    our_lines -= np.array([window_size, hypot], dtype=int)

    # crop the hough space to remove the padded windows
    our_lines = our_lines[(our_lines[:, 0] >= 0) & (our_lines[:, 0] < 180)]

    draw_lines(guide, our_lines, './outputs/front_lines.png')

    return our_lines


def draw_lines(image: np.ndarray, polar_lines: np.ndarray, filename: Optional[str] = None) -> None:
    """
    Util to draw hough lines into an image.
    :param image: image to draw the lines into
    :param polar_lines: lines to insert into the image
    :param filename: name of output file
    """
    image = image.copy()
    for theta, rho in polar_lines:
        theta *= np.pi / 180
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    ishow(image, filename=filename)
