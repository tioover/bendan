import numpy as np
from scipy import stats


class Node:
    leaf = False
    label = None
    feature = None
    branch = None

    def as_leaf(self, label):
        self.leaf = True
        self.label = label

    def as_internal(self, feature_name):
        self.leaf = False
        self.label = None
        self.feature = feature_name
        self.branch = dict()

    def add_child(self, feature_value, node):
        if self.leaf:
            raise RuntimeError("不能给叶子节点添加分支")
        self.branch[feature_value] = node


def tree_generate(features, train_set):
    """ 生成决策树 """
    node = Node()

    # 如果样本同属于一类别
    first_label = train_set[0, -1]
    if (train_set[:, -1] == first_label).all():
        # 标记为此类叶节点
        node.as_leaf(first_label)
        return node

    # 去掉示例集的标签列，获得样本集，规定最后一列是标签
    samples = train_set[:, :-1]

    # 示例集中最常见的标签
    most_frequent_label = get_most_frequent_label(train_set)

    # 如果特征集为空，或者样本集在特征集上取值相同
    if not features or (samples == samples[0]).all():
        node.as_leaf(most_frequent_label)
        return node

    #
    index, divide_train_set = best_divide_feature(features, train_set)

    # 设置当前节点为内部节点，属性值为被选中的属性
    node.as_internal(features[index])

    for feature_value in divide_train_set:
        # divide_train_set[feature_value] 为在被选中的属性上取值为 feature_value 的子集

        # 如果为空，那就创建一个叶子节点，标记为当前最多的 label
        if len(divide_train_set[feature_value]) == 0:
            child = Node()
            child.as_leaf(most_frequent_label)
        else:
            # 新的特征集，从原来的特征集中删除现在被选中的划分特征
            next_features = features[:index] + features[index:]
            child = tree_generate(next_features, divide_train_set[feature_value])
        node.add_child(feature_value, child)
    return node


def best_divide_feature(features, train_set):
    """
    根据熵算法选中一个最佳的划分属性
    返回在 features 里面的下标，和以此属性每一个值所划分的子集。
    """

    # 记录当前最大信息增益的 feature 信息
    maximum_gain = -float('inf')
    best_feature_index = None
    next_train_set = dict()  # feature 的以每个值为划分的数个子集。

    # 遍历 feature
    for i in range(len(features)):
        # 信息增益
        gain = 0.0
        divided_sets = dict()
        # 遍历当前 feature 的每个值
        for feature_value in np.unique(train_set[:, i]):
            # 以当前值划分出子集
            divided_set = np.compress(train_set[:, i] == feature_value, train_set, axis=0)
            gain -= (len(divided_set) / len(train_set)) * entropy(divided_set)
            # 记录子集
            divided_sets[feature_value] = divided_set
        if gain > maximum_gain:
            # 更新
            best_feature_index = i
            maximum_gain = gain
            next_train_set = divided_sets
    return best_feature_index, next_train_set


def entropy(examples):
    """ 计算信息熵 """
    _, counts = np.unique(examples[:, -1], return_counts=True)
    return stats.entropy(np.divide(counts, len(examples)))


def get_most_frequent_label(examples):
    """ 计算示例集中样本最多的标签 """
    labels, counts = np.unique(examples[:, -1], return_counts=True)
    return labels[np.argmax(counts)]
