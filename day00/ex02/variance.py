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



def variance(x):

	"""Computes the variance of a non-empty numpy.ndarray, using a for-loop.
	Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	The variance as a float.
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""	

	if x.size == 0:
		return None
	original_mean = mean(x)
	nb_elem = 0
	gaps_vector = np.array([])
	for elem in x:
		gap = (elem - original_mean) ** 2
		gaps_vector = np.append(gaps_vector, gap)
	return mean(gaps_vector)

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(variance(X))
print(variance(X/2))
