import numpy as np


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@np.vectorize
def d_sigmoid(y):
    return y*(1.0-y)

