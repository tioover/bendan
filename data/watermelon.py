import numpy as np

from .loader import load


# 为了方便处理，我们都默认最后一列 feature 是 label


def watermelon_2_0(remove_number=True):
    # remove_number: 是否保留编号作为属性
    number_name = "编号"
    csv_data = load("watermelon-2.0.csv")
    features = csv_data[0][:-1]
    examples = np.array(csv_data[1:])

    if remove_number:
        examples = np.delete(examples, features.index(number_name), 1)
        features.remove(number_name)

    return features, examples


def watermelon_3_0_alpha():
    csv_data = load("watermelon-3.0-alpha.csv")[1:]
    data = []
    for row in csv_data:
        if row[-1] == "1":
            label = [1.0, 0.0]
        else:
            label = [0.0, 1.0]
        data.append(list(map(float, row[:-1]))+label)
    return np.array(data)

