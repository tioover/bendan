import numpy as np
from utils import sigmoid, d_sigmoid


class NeuralNetwork:
    """ 单隐层前馈神经网络 """
    def __init__(self, learning_rate: float, input_size: int, hidden_size: int, output_size: int):
        self.eta = learning_rate
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 随机初始化参数
        self.hidden_wight = generate_random_array(input_size, hidden_size)
        self.hidden_threshold = generate_random_array(1, hidden_size)
        self.output_wight = generate_random_array(hidden_size, output_size)
        self.output_threshold = generate_random_array(1, output_size)

    def train(self, train_set):
        """ 单次训练 """
        for i in range(len(train_set)):
            self._update(train_set[i])

    def _update(self, example):
        """ 对单个示例更新参数 """
        sample = example[:-self.output_size][np.newaxis]  # 1 × input_size
        train = example[-self.output_size:][np.newaxis]  # 1 × output_size
        # b: 1 × hidden_size 隐层输出
        # y: 1 × output_size 输出层输出
        b, y = self._output(sample)
        # g: 1 × output_size 隐层 -> 输出层 参数梯度
        g = (train - y) * d_sigmoid(y)
        # e: 1 × hidden_size 输入层 -> 隐层 参数梯度
        e = b * (1-b) * (g @ np.transpose(self.output_wight))

        # 更新神经元参数
        self.output_wight += np.transpose(b) * g * self.eta
        self.output_threshold += -self.eta * g
        self.hidden_wight += self.eta * np.transpose(sample) * e
        self.hidden_threshold += -self.eta * e

    def _output(self, sample):
        b = sigmoid(sample @ self.hidden_wight - self.hidden_threshold)
        y = sigmoid(b @ self.output_wight - self.output_threshold)
        return b, y

    def output(self, sample):
        """ 获得输入样本的输出 """
        _, y = self._output(sample)
        return y[0]


def generate_random_array(*args):
    small_float = np.nextafter(0, 1)
    return np.random.uniform(small_float, 1, tuple(args))
