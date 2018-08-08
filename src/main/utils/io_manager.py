# IO MANAGER - reading in files, putting it into matrices
import numpy as np
import csv
import os

ROOT_DIRECTORY = 'MachineLearning'


def load(file_path):
    with open(res_path(file_path), newline='') as f:
        raw_data = csv.reader(f)
        return np.array([[float(val) for val in row] for row in raw_data])


def res_path(file_name):
    abs_path = os.path.abspath('')
    index = abs_path.rfind(ROOT_DIRECTORY)
    return abs_path[0:index + len(ROOT_DIRECTORY)] + '/resources/' + file_name
