import numpy as np


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@np.vectorize
def d_sigmoid(y):
    return y*(1.0-y)


def step(x):
    if x < 0.5:
        return 0.0
    elif x > 0.5:
        return 1.0
    else:
        return 0.5


vector_step = np.vectorize(step)
