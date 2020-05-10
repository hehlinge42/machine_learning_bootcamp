import numpy as np

def dot(x, y):

	"""Computes the dot product of two non-empty numpy.ndarray, using a
	for-loop. The two arrays must have the same dimensions.
	Args:
	x: has to be an numpy.ndarray, a vector.
	y: has to be an numpy.ndarray, a vector.
	Returns:
	The dot product of the two vectors as a float.
	None if x or y are empty numpy.ndarray.
	None if x and y does not share the same dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	if x.size == 0 or y.size == 0 or x.shape != y.shape:
		return None
	dot_product = 0.0
	for xi, yi in zip(x, y):
		dot_product += xi * yi
	return dot_product



def linear_mse(x, y, theta):
	
	"""Computes the mean squared error of three non-empty numpy.ndarray,
	using a for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	x: has to be an numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be an numpy.ndarray, a vector of dimension n * 1.
	Returns:
	The mean squared error as a float.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[1] != theta.shape[0] or y.shape[0] != x.shape[0] or theta.ndim != 1 or y.ndim != 1:
		return None
	summed = 0.0
	i = 0
	for line in x:
		summed += (dot(theta, line) - y[i]) ** 2
		i += 1
	return summed/x.shape[0]

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
print(linear_mse(X, Y, Z))
W = np.array([0,0,0])
print(linear_mse(X, Y, W))
