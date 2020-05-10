import numpy as np


def polynomialFeatures(x, degree=2, interaction_only= False, include_bias=True):
	"""Computes the polynomial features (including interraction terms) of a
		non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The polynomial features matrix, a numpy.ndarray.
		None if x is a non-empty numpy.ndarray.
	Raises:
		This function shouldn't raise any Exception.
	"""

	X_h = X
	if include_bias is True:
		X_h = np.insert(X_h, 0, 1., axis=1)
	
	i = 1
	while i < degree:
		i += 1
		j = 0
		while j < X.shape[1]:
			X_h = np.append(X_h, np.power(X[:,j], i).reshape(X_h.shape[0], 1), axis=1)
			#if j + 1 < X.shape[1]:
			#	X_h = np.append(X_h, np.dot(X[:,j], X[:,j + 1]), axis=1)
			j += 1
	return X_h


X = np.array([[0, 1], [2, 3], [4, 5]])
print(polynomialFeatures(X))
#[ 1., 0., 1., 0. , 0. , 1. ],
#[ 1., 2., 3., 4. , 6. , 9. ],
#[ 1., 4., 5., 16., 20., 25.]

print(polynomialFeatures(X, 3))
#[ 1., 0., 1., 0. , 0. , 1. , 0. , 0. , 0.  , 1.],
#[ 1., 2., 3., 4. , 6. , 9. , 8. , 12., 18. , 27.],
#[ 1., 4., 5., 16., 20., 25., 64., 80., 100., 125.]

print(polynomialFeatures(X, 3, interaction_only=True, include_bias=False))
#[ 0., 1., 0. ],
#[ 2., 3., 6. ],
#[ 4., 5., 20.]
