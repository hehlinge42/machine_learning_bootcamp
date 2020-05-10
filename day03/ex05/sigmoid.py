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
