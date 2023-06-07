import numpy as np
from matplotlib import pyplot as plt


def zero_pad(arr: np.ndarray, n: int = 1, mode: str = "fill") -> np.ndarray:
    """
    Pads a border of zeros around arr
    :param arr: numpy array to pad.
    :return: padded arr
    """
    if len(arr.shape) == 2:
        padd = np.zeros(np.array(arr.shape) + 2*n)
    else:
        h, w, c = arr.shape
        padd = np.zeros((h+2*n, w+2*n, c))
    padd[n:-n, n:-n] = arr = arr.copy()
    if mode == "symm":
        assert n == 1, "symmetric padding is only implemented for n=1"
        padd[0, 0] = arr[0, 0]
        padd[-1, -1] = arr[-1, -1]
        padd[0, -1] = arr[0, -1]
        padd[-1, 0] = arr[-1, 0]

        padd[1:-1, 0] = arr[:, 0]
        padd[1:-1, -1] = arr[:, -1]
        padd[0, 1:-1] = arr[0]
        padd[-1, 1:-1] = arr[-1]

    return padd


def ishow(img, filename = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    fig.show()
    if filename:
        fig.savefig(filename)