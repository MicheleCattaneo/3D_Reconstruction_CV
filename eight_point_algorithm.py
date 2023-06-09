import numpy as np
from utils import read_coords
import scipy
from DLT import DLT

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



if __name__ == '__main__':
    coords_house1 = np.array(read_coords('./coords/5coords_house1.txt'))
    coords_house2 = np.array(read_coords('./coords/5coords_house1.txt'))

    
    # fundamental matrix
    F = eight_points_algorithm(coords_house1, coords_house2)
    print('Fundamental matrix:\n',F)

