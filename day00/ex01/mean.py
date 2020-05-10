import numpy as np

def mean(x):
	"""Computes the mean of a non-empty numpy.ndarray, using a for-loop.
	Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	The mean as a float.
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""	

	if x.size == 0:
		return None
	summed = 0.0
	nb_elem = 0
	for elem in x:
		try:
			summed += elem
			nb_elem += 1
		except:
			return None
	return summed/nb_elem

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(mean(X))
print(mean(X**2))
