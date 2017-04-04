import csv
import os.path


def load(filename):
    path = os.path.split(os.path.realpath(__file__))[0]
    path = os.path.join(path, "dataset", filename)
    with open(path, encoding='utf-8') as data_file:
        reader = csv.reader(data_file)
        return list(reader)
