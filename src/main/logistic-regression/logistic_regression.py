from io_manager import *


# TODO: fix me
def map_feature(X1, X2, degree=6):
    #print(f'X1:{X1}\nX2:{X2}')
    out = np.ones((X1.shape[0], 28))
    for i in range(degree):
        for j in range(i):
            val1 = (X1 ** (i+1-j))
            val2 = (X2 ** (j))
            val3 = val1 * val2
            #print(f'val1:{val1}\nval2: {val2}\nval3:{val3}')
            #print(out[:][i+j])
            out[:, i+j+1] = val3.T
    return out


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def compute_cost(X, theta, y, lambdaa=0):
    return 1/y.shape[0] * (-y.T @ np.log(sigmoid(X @ theta))) - \
            (1 - y.T) @ np.log(1 - sigmoid(X @ theta)) + \
            lambdaa / (1/y.shape[0]) * np.sum(theta[1:])


def compute_grad(X, theta, y, lambdaa=0):
    return (1/y.shape[0] * (sigmoid(X @ theta) - y).T @ X).T + \
            lambdaa / y.shape[0] * np.concatenate(0, theta[1:])


def predict(X, theta, threshold=0.5):
    return sigmoid(X @ theta) >= threshold


if __name__ == 'builtins' or __name__ == '__main__':
    a = np.array([[1], [2], [3], [4], [5], [6], [7]], dtype=float)
    b = np.array([[10], [12], [13], [14], [15], [16], [17]], dtype=float)
    mappings = map_feature(a, b)
    print(f'OUTPUT: \n{mappings}')

