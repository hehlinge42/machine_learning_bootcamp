import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


def draw_regression(mylr):
	
	fig, ax = plt.subplots() #renvoie une figure et des axes
	ax.scatter(mylr.X, mylr.Y)	#crée un diagramme de dispersion avec X et Y
	ax.scatter(mylr.X, mylr.Y_hat, c="green")
	plt.xlabel("Quantity of blue pills (in micrograms)")
	plt.ylabel("Space driving score")
	plt.title("Evolution of the space driving score in function of the blue pill's quantity (in µg)")
	plt.plot(mylr.X, mylr.Y_hat, "--", c="green", label="model 1")
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



data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]), Xpill, Yscore)
linear_model2 = MyLR(np.array([[89.0], [-6]]), Xpill, Yscore)
linear_model1.fit_(Xpill, Yscore)
linear_model2.fit_(Xpill, Yscore)
Y_model1 = linear_model1.predict_()
Y_model2 = linear_model2.predict_()

draw_regression(linear_model1)
draw_cost_function(linear_model1)

print(linear_model1.mse_())
print(mean_squared_error(linear_model1.Y, linear_model1.Y_hat))
