# Project 2

> ### Authors:
> Michele Cattaneo
> 
> Nicolai Hermann
> 
> Oliver Tryding

## Our solution:

The goal of this project was to recover the internal and external camera parameters of the cameras that was used to take two pictures of the house scene.

To this end we manually read the pixel coordinates of the 10 marked points $x_i$ and $x_i'$ in both images and saved then in homogeneous coordinates in two files.

We then use a DLT algorithm to find the projective matrix $P$ and $P'$ such that $x = PX$ and $x' =P'X$, where $X$ are the true 3d coordinates of the points in the scene.

The rank of the matrix $A$ (such that $Ap=0$) did not have rank 11, therefore we needed to find an approximate solution by selecting $p$ to be the last column of $V$ of the singular value decomposition $A=U\Sigma V^T$. The projection matrix is then the vector $p$ re-shaped into a $3\times 4$ matrix.

We then obtained the camera calibration matrix $K$ and the camera orientation $R$ via the RQ-decomposition of $P$. We had to make sure that $K$ had positive diagonal values which was simply approached as follows:

```python
mask = np.diag(K) < 0
R[mask, :] *= -1
K[:, mask] *= -1 
```

We then did a sanity check ensuring that reconstructing $P$ using $K$ and $R$ got us back the original $P$.

Finally we tested whether the two projection matrices obtained mapped the 3d points correctly to the image points (allowing for a small error of a few pixels).


## How to run:

```shell
$ python DLT.py
```
making sure that the following files are in the folder:

-  ./coords/coords_3d.txt
- ./coords/coords_2d_house1.txt
- ./coords/coords_2d_house2.txt

The output is the following:

#### For image 1:

Calibration matrix:
```python
[[ 1.32850376e+03  2.47164506e+00  5.07139184e+02]
 [ 0.00000000e+00  1.32733446e+03  2.69518757e+02]
 [ 0.00000000e+00 -0.00000000e+00  1.00000000e+00]]
```
Camera orientation:
```python
[[ 0.80975041  0.58676479  0.00337098]
 [ 0.08895268 -0.11707439 -0.98913144]
 [-0.57999284  0.80124944 -0.14699534]]
```
$\tilde{C}$:
```python
[ 4.69680534 -6.58131086  2.01630823]
```

#### For image 2:

Calibration matrix:
```python
[[ 1.34777582e+03  3.90021104e+00  5.03374497e+02]
 [-0.00000000e+00  1.34416880e+03  2.54751651e+02]
 [-0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```
Camera orientation:
```python
[[ 4.25597842e-01  9.04912319e-01  4.13799741e-04]
 [ 3.82272245e-02 -1.75221292e-02 -9.99115436e-01]
 [-9.04104616e-01  4.25237192e-01 -4.20496667e-02]]
```
$\tilde{C}$:
```python
[ 6.40871599 -3.12310876  1.3924133 ]
```