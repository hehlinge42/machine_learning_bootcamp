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

	if theta is None or X is None or theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0]:
		print("Incompatible dimension match between X and theta.")
		return None
	X = np.insert(X, 0, 1., axis=1)
	return X.dot(theta)


def fit_(theta, X, Y, alpha = 0.001, n_cycle = 10000):
	"""
	Description:
	Performs a fit of Y(output) with respect to X.
	Args:
	theta: has to be a numpy.ndarray, a vector of dimension (number of
	features + 1, 1).
	X: has to be a numpy.ndarray, a matrix of dimension (number of
	training examples, number of features).
	Y: has to be a numpy.ndarray, a vector of dimension (number of
	training examples, 1).
	Returns:
	new_theta: numpy.ndarray, a vector of dimension (number of the
	features +1,1).
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""

	if theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0] or Y.shape[0] != X.shape[0]:
		print("Incompatible dimension match between X and theta.")
		return None
	m = X.shape[0]
	X = np.insert(X, 0, 1., axis=1)
	for i in range(n_cycle):
		hypothesis = X.dot(theta)
		parenthesis = np.subtract(hypothesis, Y)
		sigma = np.sum(np.dot(X.T, parenthesis),keepdims=True, axis=1)
		theta = theta - (alpha / m) * sigma
	return theta



X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
print(theta1)
print(predict_(theta1, X1))
X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000)
print(theta2)
print(predict_(theta2, X2))
