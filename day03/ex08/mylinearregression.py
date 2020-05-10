import numpy as np

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
    
	def __init__(self, theta, X, Y):
		"""
		Description:
		generator of the class, initialize self.
		Args:
		theta: has to be a list or a numpy array, it is a vector of
		dimension (number of features + 1, 1).
		Raises:
		This method should noot raise any Exception.
		"""
		self.theta = np.asarray(theta).reshape(len(theta), 1)
		self.X = np.asarray(X)
		self.X_h = np.insert(self.X, 0, 1., axis=1)
		self.Y = np.asarray(Y)
		self.Y_hat = None
		self.x_tests = []
		self.y_tests_pred = []
		self.y_tests_true = []


	def predict_(self, X=None, Y=None):
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
		test = True
		if X is None:
			X = self.X_h
			test = False

		if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] != self.theta.shape[0]:
			print("Dimensions are not matching")
			return None

		Y_hat = X.dot(self.theta)
		if test is False:
			self.Y_hat = Y_hat
		else:
			self.x_tests.append(X)
			self.y_tests_pred.append(Y_hat)
			self.y_tests_true.append(Y)
		return self.Y_hat



	def cost_elem_(self):
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
		
		self.Y_hat = self.predict_()
		if self.Y_hat is None:
			return None
		self.cost_elem = ((self.Y_hat - self.Y)**2)/(2*self.X.shape[0])
		return self.cost_elem


	def cost_(self):
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
		
		costs = self.cost_elem_()
		if costs is None:
			return None
		return costs.sum()



	def fit_(self, alpha = 0.0001, n_cycle = 10000):
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

		if self.theta.ndim != 2 or self.X.ndim != 2 or self.theta.shape[1] != 1 or self.X.shape[1] + 1 != self.theta.shape[0] or self.Y.shape[0] != self.X.shape[0]:
			print("Incompatible dimension match between X and theta.")
			return None
	
		m = self.X.shape[0]
		for i in range(n_cycle):
			hypothesis = self.X_h.dot(self.theta)
			parenthesis = np.subtract(hypothesis, self.Y)
			sigma = np.sum(np.dot(self.X_h.T, parenthesis),keepdims=True, axis=1)
			self.theta = self.theta - (alpha / m) * sigma
		return self.theta


	def mse_(self):

		self.predict_()
		mse = ((self.Y - self.Y_hat) ** 2).sum()
		return mse/self.X.shape[0]
