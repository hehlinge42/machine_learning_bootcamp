import numpy as np
import pandas as pd

def recall_score_(y_true, y_pred, pos_label=1):
	"""
	Compute the precision score.
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		pos_label: str or int, the class on which to report the
		precision_score (default=1)
	Returns:
		The precision score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	tp = ((y_pred == y_true) & (y_pred == pos_label)).sum()
	fn = ((y_pred != y_true) & (y_pred != pos_label)).sum()
	return tp / (tp + fn)


# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(recall_score_(y_true, y_pred))
# 0.6666666666666666


# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
'dog', 'norminet'])
print(recall_score_(y_true, y_pred, pos_label='dog'))
# 0.75


# Test n.3
print(recall_score_(y_true, y_pred, pos_label='norminet'))
# 0.5
