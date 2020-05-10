import numpy as np

def vectorized_regularization(theta, lambda_):
	"""Computes the regularization term of a non-empty numpy.ndarray, with a
	for-loop. Args:
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		lambda: has to be a float.
	Returns:
		The regularization term of theta.
		None if theta is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if (not isinstance(lambda_, (int, float))) or theta.ndim != 1:
		print(type(theta))
		return None
	return lambda_ * (theta.dot(theta.T))


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(regularization(X, 0.3))
#284.7
print(regularization(X, 0.01))
#9.47
