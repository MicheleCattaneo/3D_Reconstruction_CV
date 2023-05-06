from typing import Tuple


import numpy as np

from filters import sobel
from canny import _get_idx, filter_weak_edges


# region Angels

discrete = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]) * np.pi
get_quadrant = np.vectorize(lambda a: np.argmin(np.abs(a - discrete)) % 8)


lr = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
ud = lr.T

ul_lr = np.array([
    [-2, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 2]
])

ll_ur = np.array([
    [ 0,  1, 2],
    [-1,  0, 1],
    [-2, -1, 0]
])

i, j = 1, 1


def apply_sobel_quads_and_get_neighbours(img: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    grad_mag, alphas = sobel(img)
    quadrants = get_quadrant(alphas) % 4
    return _get_idx(quadrants[i, j], i, j, np.array([0, 0]))


def test_left_to_right():
    n1, n2 = apply_sobel_quads_and_get_neighbours(lr)
    assert n1 == (1, 2) and n2 == (1, 0), "Forward"

    n1, n2 = apply_sobel_quads_and_get_neighbours(-lr)
    assert n1 == (1, 2) and n2 == (1, 0), "Backward"


def test_up_to_down():
    n1, n2 = apply_sobel_quads_and_get_neighbours(ud)
    assert n1 == (0, 1) and n2 == (2, 1), "Forward"

    n1, n2 = apply_sobel_quads_and_get_neighbours(-ud)
    assert n1 == (0, 1) and n2 == (2, 1), "Backward"


def test_upper_left_to_lower_right():
    n1, n2 = apply_sobel_quads_and_get_neighbours(ul_lr)
    assert n1 == (0, 0) and n2 == (2, 2), "Forward"

    n1, n2 = apply_sobel_quads_and_get_neighbours(-ul_lr)
    assert n1 == (0, 0) and n2 == (2, 2), "Backward"


def test_lower_left_to_upper_right():
    n1, n2 = apply_sobel_quads_and_get_neighbours(ll_ur)
    assert n1 == (2, 0) and n2 == (0, 2), "Forward"

    n1, n2 = apply_sobel_quads_and_get_neighbours(-ll_ur)
    assert n1 == (2, 0) and n2 == (0, 2), "Backward"


# endregion

def test_weak_to_strong():
    strong = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    weak = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    expected = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 1]
    ])

    result = filter_weak_edges(strong, weak)

    assert (result == expected).all()
