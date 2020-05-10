import numpy as np

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
    
	def __init__(self, theta):
		"""
		Description:
		generator of the class, initialize self.
		Args:
		theta: has to be a list or a numpy array, it is a vector of
		dimension (number of features + 1, 1).
		Raises:
		This method should noot raise any Exception.
		"""
		self.theta = np.asarray(theta)



	def predict_(self, X):
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

		if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0]:
			print("Incompatible dimension match between X and theta.")
			return None
		X = np.insert(X, 0, 1., axis=1)
		return X.dot(self.theta)



	def cost_elem_(self, X, Y):
		"""
		Description:
		Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost
		function.
		Args:
		theta: has to be a numpy.ndarray, a vector of dimension (number of
		features + 1, 1).
		X: has to be a numpy.ndarray, a matrix of dimension (number of
		training examples, number of features).
		Returns:
		J_elem: numpy.ndarray, a vector of dimension (number of the training
		examples,1).
		None if there is a dimension matching problem between X, Y or theta.
		Raises:
		This function should not raise any Exception.
		"""
		
		Y_hat = self.predict_(X)
		if Y_hat is None:
			return None
		return ((Y_hat - Y)**2)/(2*X.shape[0])


	def cost_(self, X, Y):
		"""
		Description:
		Calculates the value of cost function.
		Args:
		theta: has to be a numpy.ndarray, a vector of dimension (number of
		features + 1, 1).
		X: has to be a numpy.ndarray, a vector of dimension (number of
		training examples, number of features).
		Returns:
		J_value : has to be a float.
		None if X does not match the dimension of theta.
		Raises:
		This function should not raise any Exception.
		"""
		
		costs = self.cost_elem_(X, Y)
		if costs is None:
			return None
		return costs.sum()



	def fit_(self, X, Y, alpha = 0.001, n_cycle = 10000):
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

		if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0] or Y.shape[0] != X.shape[0]:
			print("Incompatible dimension match between X and theta.")
			return None
	
		m = X.shape[0]
		X = np.insert(X, 0, 1., axis=1)
		for i in range(n_cycle):
			hypothesis = X.dot(self.theta)
			parenthesis = np.subtract(hypothesis, Y)
			sigma = np.sum(np.dot(X.T, parenthesis),keepdims=True, axis=1)
			self.theta = self.theta - (alpha / m) * sigma
		return self.theta
