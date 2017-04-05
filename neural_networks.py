import numpy as np
from utils import sigmoid


class NeuralNetwork:
    """ 单层前馈神经网络 """
    def __init__(self, learning_rate: float, input_size: int, hidden_size: int, output_size: int):
        self.eta = learning_rate
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_wight = generate_random_array(input_size, hidden_size)
        self.hidden_threshold = generate_random_array(hidden_size)
        self.output_wight = generate_random_array(hidden_size, output_size)
        self.output_threshold = generate_random_array(output_size)

    def train(self, train_set):
        for i in range(len(train_set)):
            self.update(train_set[i])

    def update(self, example):
        sample = example[:-self.output_size]
        b, y = self.output(sample)
        g = (example[-self.output_size:] - y) * y * (1-y)
        e = b * (1-b) * (g @ np.transpose(self.output_wight))
        tmp = self.eta * np.transpose(b) * g
        tmp.resize((self.hidden_size, self.output_size))
        self.output_wight += tmp
        self.output_threshold += -self.eta * g
        tmp = np.transpose(sample[np.newaxis]) * e
        self.hidden_wight += self.eta * tmp
        self.hidden_threshold += -self.eta * e

    def output(self, sample):
        alpha = sample @ self.hidden_wight
        b = sigmoid(alpha - self.hidden_threshold)
        beta = b @ self.output_wight
        y = sigmoid(beta - self.output_threshold)
        return b, y


def generate_random_array(*args):
    small_float = np.nextafter(0, 1)
    return np.random.uniform(small_float, 1, tuple(args))
