"""###########################################################
###                                                        ###
##    This script models the Linear Regression algorithm    ##
###                                                        ###
###########################################################"""
from io_manager import *
from plotter import *

ALPHA_VALUES = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def compute_cost(X, theta, y):
    if True or theta.shape[0] == 2:
        return ((X @ theta - y) ** 2).mean() * 1/2
    else:
        return ((X @ theta - y).T @ (X @ theta - y)) * 1/2


def compute_grad(X, theta, y, alpha, iters, debug=False):
    cost_history = []
    for i in range(iters):
        theta = theta - alpha * (1/X.shape[0]) * ((X @ theta - y).T @ X).T
        if debug:
            cost_history.append(compute_cost(X, theta, y))
    return theta if not debug else (theta, cost_history)


def feature_normalize(X):
    mu = X.mean(0)
    sigma = np.std(X, 0, ddof=1)
    return (X-mu) / sigma, mu, sigma


def predict_value(x, theta, mu, sigma):
    return ((x - mu) / sigma) @ theta


def prepare_values(data_file):
    data = load(data_file)
    X = data[:, 0:-1]
    y = data[:, -1:]
    X_norm, mu, sigma = feature_normalize(X)
    X_norm = np.concatenate((np.ones((X.shape[0], 1)), X_norm), axis=1)
    theta = np.zeros((X_norm.shape[1], 1))
    return X_norm, theta, y, mu, sigma


def normal_equation(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def automatic_linear_regression(X, theta, y, iters=50):
    min_cost = (np.inf, None, None)
    for alpha in ALPHA_VALUES:
        computed_theta, cost_history = compute_grad(X, theta, y, alpha, iters, debug=True)
        if cost_history[-1] < min_cost[0]:
            min_cost = (cost_history[-1], computed_theta, cost_history)
    return min_cost[1:] \
        if abs((min_cost[2][-1] - min_cost[2][-2]) / min_cost[2][-2]) < 1e-4 \
        else automatic_linear_regression(X, theta, y, iters*2)


def main(data_file):
    X, theta, y, mu, sigma = prepare_values(data_file)
    best_theta, cost_history = automatic_linear_regression(X, theta, y)
    print(f'RESULT ==> THETA: {best_theta.tolist()} - COST: {cost_history[-1]}')
    lineplot([i for i in range(len(cost_history))], cost_history)


if __name__ == 'builtins' or __name__ == '__main__':
    print('Program started...')
    main('data_files/ex1data2.txt')
