import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from mylinearregression import MyLinearRegression


class MyRidge(MyLinearRegression):

	def __init__(self, theta, X, Y, lambda_=1.0):
		
		MyLinearRegression.__init__(self, theta, X, Y)
		self.lambda_ = lambda_
		self.fit_x_to_h()

	def fit_x_to_h(self, X=None):

		if X is None:
			self.X_h = np.insert(self.X_h, 2, np.power(self.X_h[:,2], 3), axis=1)
			self.X_h = np.append(self.X_h, np.power(self.X_h[:,3].reshape(self.X_h.shape[0], 1), 2), axis=1)
		else:
			X = np.insert(X, 0, 1., axis=1)
			X = np.insert(X, 2, np.power(X[:,2], 3), axis=1)
			X = np.append(X, np.power(X[:,3].reshape(X.shape[0], 1), 2), axis=1)
		return X


	
	def fit_(self, lambda_=1.0, max_iter=1000, tol=0.001):
		"""
		Fit the linear model by performing Ridge regression (Tikhonov
			regularization).
		Args:
			lambda: has to be a float. max_iter: has to be integer.
			tol: has to be float.
		Returns:
			Nothing.
		Raises:
			This method should not raise any Exception.
		"""
		if (lambda_ != 1.0):
			self.lam = lambda_

		In = np.identity(self.X_h.shape[1])
		In[0][0] = 0
		parenthesis = self.X_h.transpose().dot(self.X_h) + self.lambda_ * In
		transposed = np.linalg.inv(parenthesis)
		self.theta = (transposed.dot(self.X_h.transpose())).dot(self.Y)
		self.theta = self.theta.reshape(self.theta.size, 1)

	def __str__(self):
		strn = "theta = {0}\n".format(self.theta)
		#strn += "Y_hat = {0}\n".format(self.Y_hat)
		strn += "lambda = {0}\n".format(self.lambda_)
		strn += "R2 = {0}\n".format(self.r2)
		return strn




df_train = pd.read_csv('../resources/data.csv', delimiter=',', header=1, index_col=False)

x = np.array(df_train.iloc[:, 0:2])
y = np.array(df_train.iloc[:, 2])

kfold = KFold(3, True, 1)
lam = 1.0

for train, test in kfold.split(x):

	x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

	theta = np.array([0, 0, 0, 0, 0])
	mr = MyRidge(theta, x_train, y_train, lam)
	mr.predict_()
	mr.fit_()
	mr.predict_()
	x_test = mr.fit_x_to_h(x_test)
	mr.predict_(x_test, y_test)
	mr.r2 = r2_score(mr.y_tests_true[-1], mr.y_tests_pred[-1])
	print(str(mr))
	lam += 25
