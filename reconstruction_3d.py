import numpy as np
from utils import read_coords
from eight_point_algorithm import eight_points_algorithm
import scipy

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def linear_triangulation(x, x_prime, P, P_prime):
    A = np.array([x[0]*P[2,:] - P[0,:],
                  x[1]*P[2,:] - P[1,:],
                  x_prime[0]*P_prime[2,:] - P_prime[0,:],
                  x_prime[1]*P_prime[2,:] - P_prime[1,:]
                  ])
    
    # DLT
    _, _, vh = np.linalg.svd(A)
    # last column of V contains p that minimizes Ap = 0
    X_hat = vh.T[:,-1]
    
    return X_hat

def reconstruction_3d(F, x, x_prime):
    # get left null space of F
    e_prime = scipy.linalg.null_space(F.T)
    e_prime = np.squeeze(e_prime)

    # print(e_prime)

    P = np.eye(3,4)

    skew_e = skew(e_prime)

    P_prime = np.hstack([skew_e @ F, e_prime.reshape(-1,1)])

    X_hat = []
    for i in range(x.shape[0]):
        X_hat.append(
            linear_triangulation(x[i], x_prime[i], P, P_prime) 
        )

    X_hat = np.array(X_hat)  
    return  X_hat / X_hat[:,-1].reshape(-1,1)

def DLT_homography(X, X_hat):
    n_corr = X.shape[0]

    A = np.zeros((3*n_corr, 16))

    # fill up matrix A for each correspondance 
    for i in range(n_corr):
        A[i*3, :] = np.concatenate([
            X_hat[i,-1]*X[i],
            np.zeros(4),
            np.zeros(4),
            -X_hat[i,0]*X[i]
        ])
        
        A[i*3+1, :] = np.concatenate([
            np.zeros(4),
            X_hat[i,-1]*X[i],
            np.zeros(4),
            -X_hat[i,1]*X[i]
        ])

        A[i*3+2, :] = np.concatenate([
            np.zeros(4),
            np.zeros(4),
            X_hat[i,-1]*X[i],
            -X_hat[i,2]*X[i]
        ])

    _, _, vh = np.linalg.svd(A)

    h = vh[-1]
    H = h.reshape(4,4)
    return H

if __name__ == '__main__':
    # get the 10 points on the images and 10 points in 3d
    coords_2d_house1 = np.array(read_coords('./coords/coords_2d_house1.txt'))
    coords_2d_house2 = np.array(read_coords('./coords/coords_2d_house2.txt'))
    X = np.array(read_coords('./coords/coords_3d.txt'))

    # get 5 points correspondance to get the Fundamental matrix
    coords_house1 = np.array(read_coords('./coords/5coords_house1.txt'))
    coords_house2 = np.array(read_coords('./coords/5coords_house1.txt'))
    F = eight_points_algorithm(coords_house1, coords_house2)

    X_hat = reconstruction_3d(F, coords_2d_house1, coords_2d_house2)

    H = DLT_homography(X, X_hat)
    print("Estimated Homography (H):\n", H)

    X_hat_transformed = np.dot(H, X_hat.T).T
    X_hat_transformed = X_hat_transformed / X_hat_transformed[:, -1].reshape(-1, 1)

    print("Reconstructed Points (X_hat):\n", X_hat)
    print("Transformed Reconstructed Points:\n", X_hat_transformed)

    error = np.sum(np.abs(X - X_hat_transformed))
    print("Total Error:", error)
