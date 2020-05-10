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
    if n_lab <= 1:
        return 0.0

    values, counts = np.unique(array, return_counts=True)
    probs = counts / n_lab

    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0.0

    ent = 0.0
    # Compute entropy
    base = 2
    for i in probs:
        ent -= i * log(i, base)
    return ent

array = []
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = {1, 2}
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = "bob"
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array([0, 0, 0, 0, 0, 0])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array([6])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array =  np.array(['a', 'a', 'b', 'b'])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array =  np.array(['0', '0', '1', '0', 'bob', '1'])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array =  np.array([0, 0, 1, 0, 2, 1])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array(['0', 'bob', '1'])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))

array = np.array([0., 1., 1.])
ent = entropy(array)
print("Shannon entropy for {0} is {1}".format(array, ent))
