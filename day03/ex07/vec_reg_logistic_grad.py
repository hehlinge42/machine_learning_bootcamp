import numpy as np

def sigmoid_(x, k=1, x0=0):
	"""
	Compute the sigmoid of a scalar or a list.
	Args:
	x: a scalar or list
	Returns:
	The sigmoid value as a scalar or list.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	return (1 / (1 + np.exp((-k)*(x - x0))))


def vec_reg_logistic_grad(x, y, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty
	numpy.ndarray, without any for-loop. The three arrays must have compatible
	dimensions.
	Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		alpha: has to be a float.
		lambda_: has to be a float.
	Returns:
		A numpy.ndarray, a vector of dimension n * 1, containing the results of
			the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""

	return (np.dot(x.T, (sigmoid_(np.dot(x, theta)) - y)) + lambda_ * theta) / x.shape[0]

X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
Y = np.array([1,0,1,1,1,0,0])
Z = np.array([1.2,0.5,-0.32])
print(vec_reg_logistic_grad(X, Y, Z, 1))
#array([ 6.69780169, -0.33235792, 2.71787754])
print(vec_reg_logistic_grad(X, Y, Z, 0.5))
#array([ 6.61208741, -0.3680722, 2.74073468])
print(vec_reg_logistic_grad(X, Y, Z, 0.0))
#array([ 6.52637312, -0.40378649, 2.76359183])
