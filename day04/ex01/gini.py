import numpy as np
from math import log

def gini(array):
	"""
	Computes the gini impurity of a non-empty numpy.ndarray
	:param numpy.ndarray array:
	:return float: gini_impurity as a float or None if input is not a
		non-empty numpy.ndarray
	"""
	if not isinstance(array, np.ndarray):
		return None
		
	n_lab = len(array)
	if n_lab <= 1:
		return 0.0
		
	values, counts = np.unique(array, return_counts=True)
	probs = counts / n_lab
	
	n_classes = np.count_nonzero(probs)
	if n_classes <= 1:
		return 0.0
		
	gini = 0.0
	for i in probs:
		gini += i ** 2
	return 1 - gini 

array = []
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = {1, 2}
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = "bob"
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array([0, 0, 0, 0, 0, 0])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array([6])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array =  np.array(['a', 'a', 'b', 'b'])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array =  np.array(['0', '0', '1', '0', 'bob', '1'])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array =  np.array([0, 0, 1, 0, 2, 1])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array(['0', 'bob', '1'])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))

array = np.array([0., 1., 1.])
ent = gini(array)
print("Gini impurity for {0} is {1}".format(array, ent))
