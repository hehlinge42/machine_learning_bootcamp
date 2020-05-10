import numpy as np

def sigmoid_(x, k=1, x0=0):
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

	if isinstance(x, (int, float, list)) == False and type(x) != 'numpy.ndarray':
		print("Invalid type")
		return None
	x = np.asarray(x)
	return (1 / (1 + np.exp((-k)*(x - x0))))



def log_loss_(y_true, y_pred, m, eps=1e-15):
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

	if isinstance(y_true, (int, float)) == True:
		y_true = [float(y_true)]
	if isinstance(y_pred, (int, float)) == True:
		y_pred = [float(y_pred)]
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	if y_true.shape != y_pred.shape or y_true.shape[0] != m:
		print(y_true.shape)
		print(y_pred.shape)
		print(m)
		return None
	return ((-1 / m) * (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))).sum()



# Test n.1
x = 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1 # length of y_true is 1
print(log_loss_(y_true, y_pred, m))
# 0.12692801104297152


# Test n.2
x = [1, 2, 3, 4]
y_true = 0
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
m = 1
print(log_loss_(y_true, y_pred, m))
# 10.100041078687479


# Test n.3
x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
y_true = [1, 0, 1]
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = []
for i in range(len(x_new)):
	my_sum = 0
	for j in range(len(x_new[i])):
		my_sum += x_new[i][j] * theta[j]
	x_dot_theta.append(my_sum)
y_pred = sigmoid_(x_dot_theta)
m = len(y_true)
print(log_loss_(y_true, y_pred, m))
# 7.233346147374828
