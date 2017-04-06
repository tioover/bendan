import numpy as np
from neural_networks import NeuralNetwork
from data import watermelon_3_0_alpha

data = watermelon_3_0_alpha()
nn = NeuralNetwork(0.05, 2, 3, 2)
idx = np.random.randint(len(data), size=7)
train_set = data[idx]
nn.train(train_set)
