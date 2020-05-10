import numpy as np

def predict_(theta, X):
	
	"""
	Description:
	Prediction of output using the hypothesis function (linear model).
	Args:
	theta: has to be a numpy.ndarray, a vector of dimension (number of
	features + 1, 1).
	X: has to be a numpy.ndarray, a matrix of dimension (number of
	training examples, number of features).
	Returns:
	pred: numpy.ndarray, a vector of dimension (number of the training
	examples,1).
	None if X does not match the dimension of theta.
	Raises:
	This function should not raise any Exception.
	"""

	if theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0]:
		print("Incompatible dimension match between X and theta.")
		return None
	X = np.insert(X, 0, 1., axis=1)
	return X.dot(theta)

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
print(predict_(theta1, X1))
X2 = np.array([[1], [2], [3], [5], [8]])
theta2 = np.array([[2.]])
print(predict_(theta2, X2))
X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
print(predict_(theta3, X3))
