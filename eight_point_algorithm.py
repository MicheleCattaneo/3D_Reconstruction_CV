import numpy as np
from utils import read_coords
import scipy
from DLT import DLT
import cv2
from utils import ishow

def eight_points_algorithm(x, x_prime):
    n_corr = x.shape[0]

    assert x.shape[0] == x_prime.shape[0]

    A = np.zeros((n_corr,9))

    for corr in range(n_corr):
        A[corr, :] = np.array([
            x[corr,0]*x_prime[corr,0],
            x_prime[corr,0]*x[corr,1],
            x_prime[corr,0],
            x_prime[corr,1]*x[corr,0],
            x_prime[corr,1]*x[corr,1],
            x_prime[corr,1],
            x[corr,0],
            x[corr,1],
            1.0
        ])

    # print(f'Rank of A is: {np.linalg.matrix_rank(A)}')

    _, _, vh = np.linalg.svd(A)

    f = vh.T[:,-1]

    F =  f.reshape(3,3)

    if np.linalg.matrix_rank(F) == 3:
        u, s, vh = np.linalg.svd(F)
        s[2] = 0
        F_tilde = u @ np.diag(s) @ vh

        return F_tilde

    return F

def checkF(F, x, x_prime):
    # should be 0
    for i, x in enumerate(x):
        print(x_prime[i].T @ F @ x)


if __name__ == '__main__':
    coords_house1 = np.array(read_coords('./coords/5coords_house1.txt'))
    coords_house2 = np.array(read_coords('./coords/5coords_house2.txt'))

    
    # fundamental matrix
    F = eight_points_algorithm(coords_house1, coords_house2)
    print('Fundamental matrix:\n',F)

    path = './house1.png'

    image = cv2.imread(path, 0)

    for i, point in enumerate(coords_house2):
        line = F @ point
        print(np.dot(line, coords_house2[i]))

        x_start = 0
        y_start = int((- line[0] * x_start - line[2]) / line[1])
        start_point = (y_start, x_start)

        x_end = 960
        y_end = int((- line[0] * x_end - line[2]) / line[1])
        end_point = (y_end, x_end)

        print(start_point)
        print(end_point)

        image = cv2.line(image, start_point, end_point, (0,0,0), 5)

    ishow(image, './outputs/epipolar_lines.png')
    checkF(F, coords_house1, coords_house2)