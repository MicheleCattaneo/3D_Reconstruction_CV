import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from canny import canny
from hough import guided_hough, draw_lines

from utils import zero_pad, ishow


def polar_line_to_points(line: np.ndarray):
    theta, rho = line
    theta = np.deg2rad(theta)

    x1 = rho * np.cos(theta)
    y1 = rho * np.sin(theta)

    x2 = x1 - y1
    y2 = y1 + x1

    # return x1, y1, x2, y2

    x = lambda y: (rho - y * np.sin(theta)) / np.cos(theta)

    try:
        return x(0), 0, x(1), 1
    except ZeroDivisionError:
        y = lambda x: (rho - x * np.cos(theta)) / np.sin(theta)
        return 0, y(0), 1, y(1)


def get_line_from_points(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    a = np.array([x1, y1, 1])
    # a = np.array([y1, x1, 1])
    b = np.array([x2, y2, 1])
    # b = np.array([y1, x2, 1])
    return np.cross(a, b)


def map_back_to_inf(line: np.ndarray) -> np.ndarray:
    H = np.eye(line.shape[0])
    H[-1] = line/line[-1]

    return H


def get_point_not_at_infinity(polar_line1: np.ndarray, polar_line2: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = polar_line_to_points(polar_line1)
    line1 = get_line_from_points(x1, y1, x2, y2)
    # line1 = get_line_from_points(y1, x1, y2, x2)

    x1, y1, x2, y2 = polar_line_to_points(polar_line2)
    line2 = get_line_from_points(x1, y1, x2, y2)
    # line2 = get_line_from_points(y1, x1, y2, x2)

    return np.cross(line1, line2)


def apply_mapping(image: np.ndarray, t: np.ndarray) -> np.ndarray:
    output = np.zeros_like(image)
    t_inv = np.linalg.inv(t)
    # pad = zero_pad(image, mode="symm")
    pad = image
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            pos = np.array([i, j, 1])
            pos = t_inv @ pos

            xy = pos[:2] / pos[-1]

            kl = np.round(xy).astype("int") - 1
            u, v = xy - kl - .5
            # if (kl < 0).any() or (kl >= np.array(output.shape[:-1])).any():
            #     continue

            try:
                output[j, i] = image[tuple(xy.astype(int)[::-1])]
                # k, l = kl
                # output[i, j] = np.round(
                #     (1 - v) * ((1 - u) * pad[k, l] + u * pad[k + 1, l])
                #     + v * ((1 - u) * pad[k, l + 1] + u * pad[k + 1, l + 1])
                # )
            except IndexError:
                pass

    return output


if __name__ == '__main__':
    img = np.asarray(Image.open("house1.png"))
    img = img[..., :-1]  # remove alpha channel

    # region Finding Lines

    edges = canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 10, 40)
    lines = guided_hough(edges, img, 42)

    vertical_lines = lines[lines[:, 0] < 4]
    vert1, vert2 = vertical_lines[[0, -1]]  # get first and last vertical line
    # roof1, roof2 = lines[(35 < lines[:, 0]) & (lines[:, 0] < 45)]
    # roof1[0] -= 1
    # roof1, roof2 = lines[2], lines[8]
    window_lines = lines[(90 < lines[:, 0]) & (lines[:, 0] < 100)]
    sorting = np.argsort(window_lines[:, 1])
    window1, window2 = window_lines[sorting[0]], window_lines[sorting[-1]]

    draw_lines(img.copy(), np.array([vert1, vert2, window1, window2]))

    # endregion

    # region Mapping Back

    point_not_at_infinity1 = get_point_not_at_infinity(vert1, vert2)

    point_not_at_infinity2 = get_point_not_at_infinity(window1, window2)

    line_not_at_infinity = np.cross(point_not_at_infinity1, point_not_at_infinity2)

    H_dash = map_back_to_inf(line_not_at_infinity)
    affine_image = apply_mapping(img, H_dash)

    red_pixels = np.sum(abs(img[:, :, 0] - 211) < 5)

    blue_pixels = np.sum(abs(img[:, :, 2] - 225) < 5)

    print("Number of red pixels:", red_pixels)
    print("Number of blue pixels:", blue_pixels)

    ratio = blue_pixels / red_pixels

    print(f"The size of the door is {ratio} m^2.")

    plt.imshow(affine_image)
    plt.title("Mapped back Image")
    plt.show()

    print()
    # endregion
