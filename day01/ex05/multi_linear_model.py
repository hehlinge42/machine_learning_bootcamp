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

	# Plot in function of thrust
	fig, ax = plt.subplots()
	ax.scatter(mylr.X[:,1], mylr.Y)
	ax.scatter(mylr.X[:,1], mylr.Y_hat, c="green")
	plt.xlabel("Thrust")
	plt.ylabel("Sell_price")
	plt.title("")
	fig.legend(loc="lower left")
	plt.show()
	plt.cla()

	# Plot in function of terameters
	fig, ax = plt.subplots()
	ax.scatter(mylr.X[:,2], mylr.Y)
	ax.scatter(mylr.X[:,2], mylr.Y_hat, c="pink")
	plt.xlabel("Thrust")
	plt.ylabel("Sell_price")
	plt.title("")
	fig.legend(loc="lower left")
	plt.show()
	plt.cla()



data = pd.read_csv("../resources/spacecraft_data.csv")

Y = np.array(data['Sell_price']).reshape(-1,1)
X = np.array(data['Age']).reshape(-1,1)
theta = np.array([[700.0], [-20.0]])
model_age = MyLR(theta, X, Y)
#draw_regression(model_age)

X = np.array(data['Thrust_power']).reshape(-1,1)
theta = np.array([[0.0], [40.0]])
model_thrust = MyLR(theta, X, Y)
#draw_regression(model_thrust)

X = np.array(data['Terameters']).reshape(-1,1)
theta = np.array([[800.0], [-2.0]])
model_tera = MyLR(theta, X, Y)
#draw_regression(model_tera)

X = np.array(data[['Age','Thrust_power','Terameters']])
my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], X, Y)
print(my_lreg.mse_())
my_lreg.fit_(alpha = 5e-5, n_cycle = 600000)
print(my_lreg.theta)
print(my_lreg.mse_())

draw_multi_regression(my_lreg)
