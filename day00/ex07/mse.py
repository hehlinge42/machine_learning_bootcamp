import numpy as np

def mse(y, y_hat):
	
	"""Computes the mean squared error of two non-empty numpy.ndarray, using
	a for-loop. The two arrays must have the same dimensions.
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
	summed = 0.0
	for yi, yi_hat in zip(y, y_hat):
		summed += (yi - yi_hat) ** 2
	return summed/y.size

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(mse(X, Y))
print(mse(X, X))
