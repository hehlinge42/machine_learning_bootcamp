import numpy as np

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, using
	a for-loop. The two arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a matrice of dimension m * n.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector n * 1.
	Returns:
	The gradient as a numpy.ndarray, a vector of dimensions n * 1.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[1] != theta.shape[0] or y.shape[0] != x.shape[0] or theta.ndim != 1 or y.ndim != 1:
		return None
	m = x.shape[0]
	n = x.shape[1]
	ret = np.zeros(n)
	for j in range(n):
		i = 0
		summed = 0.0
		for line in x:
			summed += (line.dot(theta) - y[i]) * x[i][j]
			i += 1
		ret[j] = summed / m
	return ret


X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
print(gradient(X, Y, Z))
W = np.array([0,0,0])
print(gradient(X, Y, W))
print(gradient(X, X.dot(Z), Z))
