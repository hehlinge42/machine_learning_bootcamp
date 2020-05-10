import numpy as np

def confusion_matrix_(y_true, y_pred, labels=None):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		labels: optional, a list of labels to index the matrix. This may be
			used to reorder or select a subset of labels. (default=None)
	Returns:
		The confusion matrix as a numpy ndarray.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if labels is None:
		labels = np.concatenate((y_true, y_pred))
		labels = np.unique(labels)
		labels = np.sort(labels)
	conf_matrix = np.zeros((len(labels), len(labels)))
	for i in range(len(labels)):
		for j in range(len(labels)):
			conf_matrix[i][j] = ((y_true == labels[i]) & (y_pred == labels[j])).sum()
	return conf_matrix


#true labels are rows and predicted labels are columns

y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog',
'bird'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y_true, y_pred))
# [[0 0 0]
# [0 2 1]
# [1 0 2]]


print(confusion_matrix_(y_true, y_pred, labels=['dog', 'norminet']))
# [[2 1]
# [0 2]]
