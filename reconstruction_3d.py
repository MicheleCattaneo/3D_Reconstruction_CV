"""
Solves Project 3
"""

from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from utils import read_coords
from eight_point_algorithm import eight_points_algorithm
import scipy
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_3D_points(X, X_prime):
    """Plots the two sets of points X and X' in 3d
    with two different colors. The points are assumed 
    to have 1s in the 4th dimension.

    Args:
        X np.ndarray: First set of points
        X_prime np.ndarray: Second set of points

    Returns:
        Figure: the figure to show.
    """
    X[:, -1] = 0
    if X_prime is not None:
        XX = np.vstack((X, X_prime))
        df = pd.DataFrame(XX, columns=['X', 'Y', 'Z', 'label'])
    else:
        df = pd.DataFrame(X, columns=['X', 'Y', 'Z', 'label'])

    fig = px.scatter_3d(df, x='X', y='Y', z='Z',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        color='label')

    return fig


def skew(x: np.ndarray) -> np.ndarray:
    """
    Returns the skew symmetric matrix for x.
    (used to compute vector matrix cross product)
    :param x: vector
    :return: skew symmetric matrix
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def linear_triangulation(x: np.ndarray, x_prime: np.ndarray,
                         P: np.ndarray, P_prime: np.ndarray) -> np.ndarray:
    """
    Computes the linear triangulation to map the two sets of (cam, point2D) to a common 3D point.
    :param x: 2D point of P
    :param x_prime: 2D point of P'
    :param P: Projection Matrix of first camera
    :param P_prime: Projection Matrix of second camera
    :return: common 3D point
    """
    A = np.array([x[0] * P[2, :] - P[0, :],
                  x[1] * P[2, :] - P[1, :],
                  x_prime[0] * P_prime[2, :] - P_prime[0, :],
                  x_prime[1] * P_prime[2, :] - P_prime[1, :]
                  ])

    # DLT
    _, _, vh = np.linalg.svd(A)
    # last column of V contains p that minimizes Ap = 0
    X_hat = vh.T[:, -1]

    return X_hat


def get_canonical_camera_pair(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the canonical camera pair based on the fundamental matrix.
    :param F: fundamental matrix (3x3)
    :return: the two cameras ((3x4), (3x4))
    """
    # get left null space of F
    e_prime = scipy.linalg.null_space(F.T)
    e_prime = np.squeeze(e_prime)

    P = np.eye(3, 4)

    skew_e = skew(e_prime)

    P_prime = np.hstack([skew_e @ F, e_prime.reshape(-1, 1)])

    return P, P_prime


def reconstruction_3d(x, x_prime, P, P_prime):
    """Given two sets of image points and the two 
    corresponding cameras, applies linear triangulation and
    returns the corresponding 3d points (world points)

    Args:
        x np.ndarray: Image 1 points
        x_prime np.ndarray: Image 2 points
        P np.ndarray: Camera matrix 1
        P_prime np.ndarray: Camera matrix 2

    Returns:
        np.ndarray: The reconstructed 3d points.
    """
    X_hat = []
    for i in range(x.shape[0]):
        X_hat.append(
            linear_triangulation(x[i], x_prime[i], P, P_prime)
        )

    X_hat = np.array(X_hat)
    return X_hat / X_hat[:, -1].reshape(-1, 1)


def DLT_homography(X_prime, X):
    """Returns a 4x4 homography matrix that transforms
     3d points X and X' such that: X = HX'

    Args:
        X_prime np.ndarray: The first set of points
        X np.ndarray: The second set of points

    Returns:
        np.ndarray: The homography matrix
    """
    n_corr = X_prime.shape[0]

    A = np.zeros((3 * n_corr, 16))

    # fill up matrix A for each correspondence
    for i in range(n_corr):
        A[i * 3, :] = np.concatenate([
            X[i, -1] * X_prime[i],
            np.zeros(4),
            np.zeros(4),
            -X[i, 0] * X_prime[i]
        ])

        A[i * 3 + 1, :] = np.concatenate([
            np.zeros(4),
            X[i, -1] * X_prime[i],
            np.zeros(4),
            -X[i, 1] * X_prime[i]
        ])

        A[i * 3 + 2, :] = np.concatenate([
            np.zeros(4),
            np.zeros(4),
            X[i, -1] * X_prime[i],
            -X[i, 2] * X_prime[i]
        ])

    _, _, vh = np.linalg.svd(A)

    h = vh[-1]
    H = h.reshape(4, 4)
    return H


def check_reconstruction(X, X_prime, eps=0.1):
    """Checks that two sets of corresponding points
    are all close to each other within an eps difference

    Args:
        X np.ndarray: first set of points
        X_prime np.ndarray: second set of points
        eps (float, optional): The maximal allowed difference on each dimension. Defaults to 0.1.
    """
    for a, b in zip(X, X_prime):
        assert (np.abs(a - b) < eps).all()


def connect_points(edges: List[Tuple[int, int]], X: np.ndarray, fig: plt.Figure) -> None:
    """
    Connects points in a scatter plot figure.
    :param edges: list of points to connect.
    :param X: 3d points
    :param fig: figure of the plot.
    """
    for e in edges:
        e_start, e_end = e
        fig.add_trace(go.Scatter3d(
            x=[X[e_start][0], X[e_end][0]],
            y=[X[e_start][1], X[e_end][1]],
            z=[X[e_start][2], X[e_end][2]],
            mode='lines'
        ))


if __name__ == '__main__':
    # get the 10 points on the images and 10 points in 3d
    coords_2d_house1 = np.array(read_coords('./coords/coords_2d_house1.txt'))
    coords_2d_house2 = np.array(read_coords('./coords/coords_2d_house2.txt'))
    X = np.array(read_coords('./coords/coords_3d.txt'))

    F = eight_points_algorithm(coords_2d_house1, coords_2d_house2)

    # Get canonical pair of cameras
    P, P_prime = get_canonical_camera_pair(F)
    # Reconstruct 3d points using the canonical pair
    X_hat = reconstruction_3d(coords_2d_house1, coords_2d_house2, P, P_prime)

    # Get the homography that relates the canonical pair to the correct pair of cameras
    H = DLT_homography(X_hat, X)
    # print("Estimated Homography (H):\n", H)

    # Apply H on the points 
    X_hat_transformed = np.dot(H, X_hat.T).T
    X_hat_transformed = X_hat_transformed / X_hat_transformed[:, -1].reshape(-1, 1)

    error = np.sum(np.abs(X - X_hat_transformed))
    print("Total Error:", error)

    # plot the reconstruction
    check_reconstruction(X, X_hat_transformed, eps=0.1)
    fig = plot_3D_points(X, X_hat_transformed)
    connect_points([(1, 2), (6, 8),
                    (6, 3), (3, 4)], X, fig)
    fig.show()

    # Get the true camera pairs by applying the inverse of the homography H
    P_tilde = P @ np.linalg.inv(H)
    P_prime_tilde = P_prime @ np.linalg.inv(H)

    # Read the unknown 5 points 
    unknown_5_points1 = np.array(read_coords('./coords/5coords_house1.txt'))
    unknown_5_points2 = np.array(read_coords('./coords/5coords_house2.txt'))
    # reconstruct their 3d coordinates
    X5 = reconstruction_3d(unknown_5_points1, unknown_5_points2, P_tilde, P_prime_tilde)

    # add the five unknown points to the scatter plot.
    fig2 = plot_3D_points(X, X5)
    connect_points([
        (1, 2), (6, 8), (6, 3), (6, 14),
        (3, 4), (4, 14), (3, 14), (6, 4),
        (1, 12), (7, 13), (5, 10)
    ], np.vstack([X, X5]), fig2)
    fig2.show()
