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
		self.X_with1 = np.insert(self.X, 0, 1., axis=1)
		self.Y = np.asarray(Y)
		self.Y_hat = None


	def predict_(self):
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

		if self.theta.ndim != 2 or self.X.ndim != 2 or self.theta.shape[1] != 1 or self.X.shape[1] + 1 != self.theta.shape[0]:
			print('Shape of x: {0}'.format(self.X.shape))
			print('Shape of theta: {0}'.format(self.theta.shape))
			print("Incompatible dimension match between X and theta.")
			return None
		self.Y_hat = self.X_with1.dot(self.theta)
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
			hypothesis = self.X_with1.dot(self.theta)
			parenthesis = np.subtract(hypothesis, self.Y)
			sigma = np.sum(np.dot(self.X_with1.T, parenthesis),keepdims=True, axis=1)
			self.theta = self.theta - (alpha / m) * sigma
		return self.theta


	def mse_(self):

		self.predict_()
		mse = ((self.Y - self.Y_hat) ** 2).sum()
		return mse/self.X.shape[0]



	def normalequation_(self):
		"""
		Description:
		Perform the normal equation to get the theta parameters of the
		hypothesis h and stock them in self.theta.
		Args:
		X: has to be a numpy.ndarray, a matrix of dimension (number of
		training examples, number of features)
		Y: has to be a numpy.ndarray, a vector of dimension (number of
		training examples,1)
		Returns:
		No return expected.
		Raises:
		This method should not raise any Exceptions.
		"""

		parenthesis = self.X_with1.transpose().dot(self.X_with1)
		transposed = np.linalg.inv(parenthesis)
		self.theta = (transposed.dot(self.X_with1.transpose())).dot(self.Y)
		print(self.theta)
		return self.theta
