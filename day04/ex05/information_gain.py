import numpy as np
from math import log

def entropy(array):
    """
    Computes the Shannon Entropy of a non-empty numpy.ndarray
    :param numpy.ndarray array:
    :return float: shannon's entropy as a float or None if input is not a
		non-empty numpy.ndarray
    """

    if not isinstance(array, np.ndarray):
        return None

    n_lab = len(array)
    if n_lab <= 0:
        return None

    values, counts = np.unique(array, return_counts=True)
    probs = counts / n_lab

    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return None

    ent = 0.0
    # Compute entropy
    base = 2
    for i in probs:
        ent -= i * log(i, base)
    return ent



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
	if n_lab <= 0:
		return None
		
	values, counts = np.unique(array, return_counts=True)
	probs = counts / n_lab
	
	n_classes = np.count_nonzero(probs)
	if n_classes <= 1:
		return None
		
	gini = 0.0
	for i in probs:
		gini += i ** 2
	return 1 - gini
	
	
def information_gain(array_source, array_children_list, criterion='gini'):
	"""
	Computes the information gain between the first and second array using
	the criterion ('gini' or 'entropy')
	:param numpy.ndarray array_source:
	:param list array_children_list: list of numpy.ndarray
	:param str criterion: Should be in ['gini', 'entropy']
	:return float: Shannon entropy as a float or None if input is not a
	non-empty numpy.ndarray or None if invalid input
	""" 
	if criterion not in ('gini', 'entropy'):
		return None
	if criterion == 'gini':
		s0 = gini(array_source)
		s1 = gini(array_children_list)
	else:
		s0 = entropy(array_source)
		s1 = entropy(array_children_list)
	if s1 is None or s0 is None:
		return None
	return s0 - s1

array_source = np.array([])
array_children = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
print("Information gain between {0} and {1} is {2} with criterion 'gini' and {3} with criterion 'entropy'".format(array_source, array_children, information_gain(array_source, array_children, 'gini'), information_gain(array_source, array_children, 'entropy')))

array_source = ['a' 'a' 'b' 'b']
array_children = {1, 2}
print("Information gain between {0} and {1} is {2} with criterion 'gini' and {3} with criterion 'entropy'".format(array_source, array_children, information_gain(array_source, array_children, 'gini'), information_gain(array_source, array_children, 'entropy')))


array_source = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
array_children = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
print("Information gain between {0} and {1} is {2} with criterion 'gini' and {3} with criterion 'entropy'".format(array_source, array_children, information_gain(array_source, array_children, 'gini'), information_gain(array_source, array_children, 'entropy')))


array_source = np.array(['0', '0', '1', '0', 'bob', '1'])
array_children = np.array([0, 0, 1, 0, 2, 1])
print("Information gain between {0} and {1} is {2} with criterion 'gini' and {3} with criterion 'entropy'".format(array_source, array_children, information_gain(array_source, array_children, 'gini'), information_gain(array_source, array_children, 'entropy')))
