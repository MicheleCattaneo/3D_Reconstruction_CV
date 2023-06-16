import numpy as np
from utils import read_coords
import scipy
from DLT import DLT
import cv2
from utils import ishow


def eight_points_algorithm(x, x_prime):
    n_corr = x.shape[0]

    assert x.shape[0] == x_prime.shape[0]

    A = np.zeros((n_corr, 9))

    for corr in range(n_corr):
        A[corr, :] = np.array([
            x[corr, 0] * x_prime[corr, 0],
            x_prime[corr, 0] * x[corr, 1],
            x_prime[corr, 0],
            x_prime[corr, 1] * x[corr, 0],
            x_prime[corr, 1] * x[corr, 1],
            x_prime[corr, 1],
            x[corr, 0],
            x[corr, 1],
            1.0
        ])

    # print(f'Rank of A is: {np.linalg.matrix_rank(A)}')

    _, _, vh = np.linalg.svd(A)

    f = vh.T[:, -1]

    F = f.reshape(3, 3)

    if np.linalg.matrix_rank(F) == 3:
        u, s, vh = np.linalg.svd(F)
        s[2] = 0
        F_tilde = u @ np.diag(s) @ vh

        return F_tilde

    return F


def checkF(F, x, x_prime, eps=0.5):
    # should be 0
    for i, x in enumerate(x):
        assert np.abs(x_prime[i].T @ F @ x - 0) < eps


def drawlines(img, lines):
    """Given an image, draws the given epipolar lines on the image and returns it.

    Args:
        img np.ndarray: The image/figure.
        lines np.ndarray: An array of epipolar lines (a,b,c) such that they describe a line
        ax + by + c = 0

    Returns:
        np.ndarray: The image with the lines drawn on it.
    """
    r, c = img.shape[0], img.shape[1]

    for r in lines:
        color = (0, 0, 0)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)

    return img


def draw_points(img, points, labels=True):
    color = (0, 0, 255)
    for i, p in enumerate(points):
        img = cv2.circle(img, tuple(map(int, p)), 5, color, -1)
        if labels:
            img = cv2.putText(img, f'{i + 1}', org=tuple(map(int, p)),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              color=(0, 0, 0),
                              thickness=2,
                              lineType=2)

    return img


def computeEpipolarLines(points, F, image=1) -> np.ndarray:
    """Computes the epipolar lines as arrays (a,b,c) for each point in points
    using the fundamental matrix F. 
    If image==1, then the given points are points from image 1 and the lines are computed
    for image 2 as l' = Fx.
    If image==2, then the given points are points from image 2 and the lines are computed 
    for image 1 as l = F^Tx'.

    Args:
        points np.ndarray: The points on either image 1 or image 2.
        F np.ndarray: The fundamental matrix
        image (int, optional): Defines from which image the given points are coming from. It can either be
        1 of 2. If 1 is given, the lines returned are for image 2. If 2 is given the lines returned are for
        image 1. Defaults to 1.

    Returns:
        np.ndarray: A np.ndarray of shape (n,3) containing the epipolar lines (a,b,c) for each given point.
    """
    n = points.shape[0]
    lines = []
    for p in points:
        if image == 1:
            line = F @ p
        else:
            line = F.T @ p
        lines.append(line)

    return np.array(lines)


if __name__ == '__main__':
    # coords_house1 = np.array(read_coords('./coords/5coords_house1.txt'))
    # coords_house2 = np.array(read_coords('./coords/5coords_house2.txt'))

    ten_coords_2d_house1 = np.array(read_coords('./coords/coords_2d_house1.txt'))
    ten_coords_2d_house2 = np.array(read_coords('./coords/coords_2d_house2.txt'))

    # fundamental matrix
    F = eight_points_algorithm(ten_coords_2d_house1, ten_coords_2d_house2)
    # F_true = cv2.findFundamentalMat(ten_coords_2d_house1, ten_coords_2d_house2)[0]

    print('Fundamental matrix:\n', F)

    # print('True F: \n', F_true)
    path1 = './house1.png'
    path2 = './house2.png'

    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    epipolar_lines1 = computeEpipolarLines(ten_coords_2d_house2, F, image=2)
    image1 = drawlines(image1, epipolar_lines1)
    image1 = draw_points(image1, ten_coords_2d_house1[:, :-1])

    epipolar_lines2 = computeEpipolarLines(ten_coords_2d_house1, F, image=1)
    image2 = drawlines(image2, epipolar_lines2)
    image2 = draw_points(image2, ten_coords_2d_house2[:, :-1])

    ishow(image1, './outputs/epipolar_lines1.png')
    ishow(image2, './outputs/epipolar_lines2.png')
    checkF(F, ten_coords_2d_house1, ten_coords_2d_house2, eps=0.5)
    # checkF(F_true, ten_coords_2d_house1, ten_coords_2d_house2)
