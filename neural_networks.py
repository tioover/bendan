import numpy as np
from utils import sigmoid


class NeuralNetwork:
    """ 单层前馈神经网络 """
    def __init__(self, learning_rate: float, input_size: int, hidden_size: int, output_size: int):
        self.eta = learning_rate
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_wight = generate_random_array(input_size, hidden_size)
        self.hidden_threshold = generate_random_array(1, hidden_size)
        self.output_wight = generate_random_array(hidden_size, output_size)
        self.output_threshold = generate_random_array(1, output_size)

    def train(self, train_set):
        for i in range(len(train_set)):
            self.update(train_set[i])

    def update(self, example):
        sample = example[:-self.output_size][np.newaxis]  # input_size x 1
        train = example[-self.output_size:][np.newaxis]  # output_size x 1
        b, y = self.output(sample)
        g = (train - y) * y * (1-y)  # 隐层 -> 输出层 参数梯度
        e = b * (1-b) * (g @ np.transpose(self.output_wight))  # 输入层 -> 隐层 参数梯度

        # 更新神经元参数
        self.output_wight += np.transpose(b) * g * self.eta
        self.output_threshold += -self.eta * g
        self.hidden_wight += self.eta * np.transpose(sample) * e
        self.hidden_threshold += -self.eta * e

    def output(self, sample):
        b = sigmoid(sample @ self.hidden_wight - self.hidden_threshold)
        y = sigmoid(b @ self.output_wight - self.output_threshold)
        return b, y


def generate_random_array(*args):
    small_float = np.nextafter(0, 1)
    return np.random.uniform(small_float, 1, tuple(args))
