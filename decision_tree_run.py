from data import watermelon_2_0
from decision_tree import tree_generate


def get_tree():
    features, examples = watermelon_2_0()
    return tree_generate(features, examples)


def make_dot_source(root):
    def node_source(node):
        dot_label = node.feature
        if node.leaf:
            dot_label = node.label
        result = "%d [label=\"%s\"];\n" % (id(node), dot_label)
        if node.leaf:
            return result
        for feature_value in node.branch:
            child = node.branch[feature_value]
            result += "%d -> %d [label=\"%s\"];\n" % (id(node), id(child), feature_value)
            result += node_source(child)
        return result

    source = "digraph \"Decision tree\" {\n"
    source += node_source(root)
    source += "}\n"
    return source


def main():
    root = get_tree()
    dot_source = make_dot_source(root)
    with open("decision_tree.dot", "w", encoding="utf-8") as dot:
        dot.write(dot_source)


if __name__ == '__main__':
    main()
