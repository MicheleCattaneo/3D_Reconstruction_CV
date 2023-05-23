from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import zero_pad, ishow


def filter(image: np.ndarray, edges: np.ndarray, colors: List[np.ndarray]):
    mask = np.zeros_like(edges, dtype=bool)
    padded_img = zero_pad(image, 3)
    dist = lambda M, x: np.sum(np.abs(M - x), axis=2)

    for i, j in np.argwhere(edges):
        dists_to_colors = list(map(lambda c: np.min(dist(padded_img[i:i + 7, j:j + 7], c)), colors))

        mask[i, j] = np.sum(np.array(dists_to_colors) < 12) > 1
    return mask


def get_maxima(img: np.ndarray, threshold: int, window: int = 10) -> np.ndarray:
    mask = img > threshold
    padded = zero_pad(img, window)

    for i, j in np.argwhere(mask):
        mask[i, j] = img[i, j] >= padded[i:i+2*window+1, j:j+2*window+1].max()

    return np.argwhere(mask)


def polar_to_line(polar, img):
    intersections = np.full((polar.shape[0], polar.shape[0], 2), np.nan)

    for i, (theta1, rho1) in enumerate(polar):
        for j, (theta2, rho2) in enumerate(polar[:i+1]):
            b = np.cos(theta1)
            a = np.sin(theta2)
            c = rho1

            e = np.cos(theta2)
            d = np.sin(theta2)
            f = rho2

            try:
                x = (c*e - b*f)/(a*e - b*d)
                y = (a*f - c*d)/(a*e - b*d)

                intersections[i, j] = np.array([x, y])
            except ZeroDivisionError:
                pass
    intersections = intersections

    # inte = np.nan_to_num(intersections, nan=-1, posinf=-1, neginf=-1)
    # points = intersections[(intersections < np.array([540, 2204])).all(axis=2) & (0 <= intersections).all(axis=2)]

    for row in intersections:
        for i in row:
            if np.isnan(i).any() or np.isinf(i).any():
                continue

            # if (i > 0).all() and (i < np.array([2204, 540])).all():
            cv2.circle(img, i.round().astype(int), 10, (255, 255, 0))

    # plt.scatter(points[:, 1], points[:, 0], alpha=.4)
    # plt.gca().invert_yaxis()
    # plt.show()
    ishow(img)
    print()


def guided_hough(
        edges: np.ndarray, guide: np.ndarray,
        voting_threshold: int
):
    mask = filter(guide, edges, [
        # np.array([163, 194, 205]),
        np.array([0, 0, 224]),
        np.array([235, 237, 238]),
        # np.array([102, 116, 124]),
        np.array([85, 146, 197]),
        np.array([211, 0, 0]),
        np.array([43, 108, 172])
    ])

    # mask = np.eye(11, 11)[::-1]

    window_size = 10

    plt.imshow(mask, cmap="gray")
    plt.show()
    hypot = np.ceil(np.hypot(*mask.shape))
    H = np.zeros((180+2*window_size, int(hypot*2)))
    for i, j in np.argwhere(mask):
        for theta in range(-window_size, 180+window_size):
            theta_rad = theta * np.pi/180
            p = np.cos(theta_rad) * j + np.sin(theta_rad) * i
            H[theta+window_size, int(np.round(p) + hypot)] += 1
    our_lines = get_maxima(H, voting_threshold, window_size)
    our_lines -= np.array([window_size, hypot], dtype=int)
    our_lines = our_lines[(our_lines[:, 0] >= 0) & (our_lines[:, 0] < 180)]


    draw_lines(guide, our_lines)

    return our_lines


def draw_lines(image, polar_lines):
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
    ishow(image)
