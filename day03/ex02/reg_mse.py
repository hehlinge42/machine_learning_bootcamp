import numpy as np

def regularization(theta, lambda_):
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

def reg_mse(x, y, theta, lambda_):
	"""Computes the regularized mean squared error of three non-empty
	numpy.ndarray, without any for-loop. The three arrays must have compatible
	dimensions.
	Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		lambda: has to be a float.
	Returns:
		The mean squared error as a float.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	dot_term = (x.dot(theta) - y)
	return (np.dot(dot_term.T, dot_term) + regularization(theta, lambda_)) / (x.shape[0])

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
print(reg_mse(X, Y, Z, 0))
#2641.0
print(reg_mse(X, Y, Z, 0.1))
#2641.6464285714287
print(reg_mse(X, Y, Z, 0.5))
#2644.2321428571427
