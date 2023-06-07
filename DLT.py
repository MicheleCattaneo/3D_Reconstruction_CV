from utils import read_coords
import numpy as np

def DLT(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_corr = X.shape[0]

    assert X.shape[0] == x.shape[0]

    A = np.zeros((2*n_corr, 12))

    # fill up matrix A for each correspondance 
    for i in range(n_corr):
        A[i*2, 4:] = np.concatenate([
            -x[i,-1]*X[i], 
            x[i,1]*X[i]
            ])
        
        A[i*2+1, :] = np.concatenate([
            x[i,-1]*X[i],
            np.zeros(4),
            -x[i,0]*X[i]
        ])

    # could avoid SVD decomp if rank is 11
    rank = np.linalg.matrix_rank(A)

    _, _, vh = np.linalg.svd(A)
    # last column of V contains p that minimizes Ap = 0
    p = vh.T[:,-1]
    # reshape it into the projective matrix P
    P = p.reshape(3,4)
    return P


def get_camera_parameters(P: np.ndarray):
    M = P[:,:-1]

    # Get K and R with a RQ decomposition.
    # permutation matrix with 1s on the antidiagonal
    permut = np.fliplr(np.eye(3))

    q, r = np.linalg.qr(M.T@permut)

    # calibration matrix 
    K = permut @ r.T @ permut
    K /= K[-1,-1]
    # camera orientation 
    R = permut @ q.T

    C_tilde = np.linalg.inv(-M) @ P[:,-1]

    # make sure K has positive diagonal values and adapt R
    mask = np.diag(K) < 0
    R[mask, :] *= -1
    K[:, mask] *= -1 

    # P_hat = K @ (np.hstack([R, (-R@C_tilde).reshape(-1,1)]))
    P_hat = K @ R  @ (np.hstack([np.eye(3), (-C_tilde).reshape(-1,1)]))
    P_hat_norm = P_hat / P_hat[-1,-1]
    P_norm = P / P[-1,-1]

    assert np.isclose(P_norm, P_hat_norm).all()

    return K, R, C_tilde


if __name__ == '__main__':
    coords_3d = np.array(read_coords('./coords/coords_3d.txt'))
    coords_2d_house1 = np.array(read_coords('./coords/coords_2d_house1.txt'))

    P = DLT(coords_3d, coords_2d_house1)
    K, R, C_tilde = get_camera_parameters(P)

