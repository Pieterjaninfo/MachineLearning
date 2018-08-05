"""###########################################################
###                                                        ###
##    This script models the Linear Regression algorithm    ##
###                                                        ###
###########################################################"""
from io_manager import *


def compute_cost(X, theta, y):
    return ((X.dot(theta) - y) ** 2).mean() * 1/2


def main(data_file):
    data = load(data_file)
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:, 0:-1]), axis=1)
    y = data[:, -2:-1]
    theta = np.zeros((X.shape[1], 1))

    print(X)
    print(y)
    print(theta)

    print(X.shape, y.shape, theta.shape)
    print(compute_cost(X, theta, y))


if __name__ == 'builtins':
    print('Program started...')
    main('data_files/ex1data1.txt')
