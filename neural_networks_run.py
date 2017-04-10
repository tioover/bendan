import numpy as np
from neural_networks import NeuralNetwork

xor = np.array(
    [[1.0, 0.0, 1.0],
     [0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0],
     [0.0, 0.0, 0.0]]
)


def xor_network():
    nn = NeuralNetwork(0.05, 2, 4, 1)
    for _ in range(10000):
        nn.train(xor)
    return nn


if __name__ == '__main__':
    xor_network()
