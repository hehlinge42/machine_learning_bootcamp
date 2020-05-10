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

	if x.size == 0 or y.size == 0 or x.size != y.size:
		return None
	dot_product = 0.0
	for xi, yi in zip(x, y):
		dot_product += xi * yi
	return dot_product

def mat_mat_prod(x, y):
	"""Computes the product of two non-empty numpy.ndarray, using a
	for-loop. The two arrays must have compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a matrix of dimension m * n.
	y: has to be an numpy.ndarray, a vector of dimension n * p.
	Returns:
	The product of the matrices as a matrix of dimension m * p.
	None if x or y are empty numpy.ndarray.
	None if x and y does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	if x.shape[1] != y.shape[0]:
		return None
	ret = np.array([])
	for line in x:
		for column in y.T:
			dot_prod = dot(line, column)
			ret = np.append(ret, dot_prod)
	return ret.reshape(x.shape[0], y.shape[1])

W = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14],
[-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13],
[ 2, -1, 12, 3, -7, -3, -6]])
Z = np.array([
[ -6, -1, -8, 7, -8],
[ 7, 4, 0, -10, -10],
[ 7, -13, 2, 2, -11],
[ 3, 14, 7, 7, -4],
[ -1, -3, -8, -4, -14],
[ 9, -14, 9, 12, -7],
[ -9, -4, -10, -3, 6]])
print(mat_mat_prod(W, Z))
print(mat_mat_prod(Z,W))
