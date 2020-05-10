import numpy as np
import pandas as pd


class LogisticRegressionBatchGd:

	def __init__(self, alpha=0.001, max_iter=1000, verbose=True, learning_rate='constant'):
		self.alpha = alpha
		self.max_iter = max_iter
		self.verbose = verbose
		self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
		self.threshold = 0.5
		self.thetas = []
		self.costs = []


	def __sigmoid(self, x, k=1, x0=0):
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

	
	def __log_loss(self, y_true, y_pred, eps=1e-15):
		"""
		Compute the logistic loss value.
		Args:
			y_true: a scalar or a list for the correct labels
			y_pred: a scalar or a list for the predicted labels
		m: the length of y_true (should also be the length of y_pred)
		eps: epsilon (default=1e-15)
		Returns:
			The logistic loss value as a float.
			None on any error.
		Raises:
			This function should not raise any Exception.
		"""
		m = y_true.shape[0]
		if y_true.shape != y_pred.shape or y_true.shape[0] != m:
			return None
		return ((-1 / m) * (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))).sum()


	def __log_gradient(self, x, y_true, y_pred):
		"""
		Compute the gradient.
		Args:
			x: a list or a matrix (list of lists) for the samples
			y_true: a scalar or a list for the correct labels
			y_pred: a scalar or a list for the predicted labels
		Returns:
			The gradient as a scalar or a list of the width of x.
			None on any error.
		Raises:
		This function should not raise any Exception.
		"""

		x = np.array(x)
		if x.ndim == 1:
			x = np.array(x).reshape(1, len(x))
		return (y_pred - y_true).dot(x)


	def cost(self, x_train, y_train):
		"""
		Appends the result of the cost function computed with
			the last elem of the thetas list.
		Arg:
			x_train: a 1d or 2d numpy ndarray for the samples
			y_train: a scalar or a numpy ndarray for the correct labels
		Returns:
			Mean accuracy of self.predict(x_train) with respect to y_true
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		y_predict = self.predict(x_train)
		self.costs.append(self.__log_loss(y_train, y_predict))


	def fit(self, x_train, y_train):
		"""
		Fit the model according to the given training data.
		Args:
			x_train: a 1d or 2d numpy ndarray for the samples
			y_train: a scalar or a numpy ndarray for the correct labels
		Returns:
			self : object
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""

		m = x_train.shape[0] # Nb of training examples
		n = x_train.shape[1] # Nb of features

		gap = int(self.max_iter / 10)
		self.thetas = np.zeros(n)
		self.cost(x_train, y_train)
		for i in range(self.max_iter):
			hypothesis = self.__sigmoid(x_train.dot(self.thetas))
			gradient = self.__log_gradient(x_train, y_train, hypothesis)
			self.thetas = (self.thetas - (self.alpha / m) * gradient)
			self.cost(x_train, y_train)
		return self


	
	def predict(self, x_train):
		"""
		Predict class labels for samples in x_train.
		Arg:
			x_train: a 1d or 2d numpy ndarray for the samples
		Returns:
			y_pred, the predicted class label per sample.
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		return self.__sigmoid(x_train.dot(self.thetas)) >= self.threshold



	def score(self, x_train, y_train):
		"""
		Returns the mean accuracy on the given test data and labels.
		Arg:
			x_train: a 1d or 2d numpy ndarray for the samples
			y_train: a scalar or a numpy ndarray for the correct labels
		Returns:
			Mean accuracy of self.predict(x_train) with respect to y_true
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		y_pred = self.predict(x_train)
		print(y_pred)
		print(y_train)
		return (y_pred == y_train).mean()
		

# To download the file, please click on the link provided in the 'resource_links.txt'
# And change the path of the read_csv function accordingly

df_train = pd.read_csv('../../../dataset/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)


x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv('../../../dataset/test_dataset_clean.csv', delimiter=',', header=None,
index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]

# We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
model = LogisticRegressionBatchGd(alpha=0.01, max_iter=1500, verbose=True, learning_rate='constant')
# We fit our model to our dataset and display the score for the train and test datasets
model.fit(x_train, y_train)
print(f'Score on train dataset : {model.score(x_train, y_train)}')
y_pred = model.predict(x_test)
print(f'Score on test dataset : {(y_pred == y_test).mean()}')
