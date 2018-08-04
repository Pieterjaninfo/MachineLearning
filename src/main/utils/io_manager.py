# IO MANAGER - reading in files, putting it into matrices
import numpy as np
import csv


def load(file_path):
    with open(res_path(file_path), newline='') as f:
        raw_data = csv.reader(f)
        return np.array([[float(val) for val in row] for row in raw_data])


def res_path(file_name):
    return '../../../resources/' + file_name
