import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


def draw_regression(mylr, x_label="", y_label="", fig_title="", legend="model"):
	
	mylr.fit_()
	mylr.predict_()

	fig, ax = plt.subplots() #renvoie une figure et des axes
	ax.scatter(mylr.X, mylr.Y)	#crée un diagramme de dispersion avec X et Y
	ax.scatter(mylr.X, mylr.Y_hat, c="green")
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(fig_title)
	plt.plot(mylr.X, mylr.Y_hat, "--", c="green", label="model")
	fig.legend(loc="lower left")
	plt.show()
	plt.cla()



def draw_cost_function(mylr):

	fig, ax = plt.subplots() #renvoie une figure et des axes
	t0 = mylr.theta[0]
	thetas_0 = [t0 - 20, t0 - 10, t0, t0 + 10, t0 + 20]
	for theta_0 in thetas_0:
		theta = np.linspace(-14, 4, 100).reshape(100, 1)
		theta = np.insert(theta, 0, theta_0, axis=1)
		y = np.array([])
		for i in range(theta.shape[0]):
			tmp_lr = MyLR(theta[i].reshape(2, 1), mylr.X, mylr.Y) 
			dot = tmp_lr.cost_()
			y = np.append(y, dot)
		plt.plot(theta[:,1], y)

	plt.xlabel("Theta1")
	plt.ylabel("Cost function J(Theta0, Theta1")
	plt.title("Evolution of the cost function J in fuction of Theta0 for different values of Theta1")
	plt.show()
	plt.cla()


def draw_multi_regression(mylr):
	
	#mylr.fit_()
	mylr.predict_()

	# Plot in function of age
	fig, ax = plt.subplots()
	ax.scatter(mylr.X[:,0], mylr.Y)
	ax.scatter(mylr.X[:,0], mylr.Y_hat, c="blue")
	plt.xlabel("Age")
	plt.ylabel("Sell_price")
	plt.title("")
	fig.legend(loc="lower left")
	plt.show()
	plt.cla()



data = pd.read_csv("../resources/spacecraft_data.csv")

Y = np.array(data['Sell_price']).reshape(-1,1)
X = np.array(data[['Age','Thrust_power','Terameters']])

myLR_ne = MyLR([1., 1., 1., 1.], X, Y)
myLR_lgd = MyLR([1., 1., 1., 1.], X, Y)
myLR_lgd.fit_(alpha = 5e-5, n_cycle = 10000)
myLR_ne.normalequation_()
print(myLR_lgd.mse_())
print(myLR_ne.mse_())

draw_multi_regression(myLR_ne)
