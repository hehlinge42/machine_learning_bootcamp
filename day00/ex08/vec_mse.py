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



def vec_mse(y, y_hat):
	"""Computes the mean squared error of two non-empty numpy.ndarray,
	without any for loop. The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector.
	Returns:
	The mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.ndarray.
	None if y and y_hat does not share the same dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	if y.shape != y_hat.shape or y.ndim != 1:
		return None
	return dot(y_hat - y, y_hat - y)/y.size

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(vec_mse(X, Y))
print(vec_mse(X, X))
