# test the IO manager file
from matrix_operations import *
import unittest
import numpy as np
from linear_regression import prepare_values, compute_grad, compute_cost, normal_equation


class TestLinReg(unittest.TestCase):
    def setUp(self):
        self.val = prepare_values('data_files/ex1data2.txt')
        self.x = self.val[0].tolist()
        self.theta = self.val[1].tolist()
        self.y = self.val[2].tolist()
        self.mu = self.val[3].tolist()
        self.sigma = self.val[4].tolist()
        self.best_theta = compute_grad(self.val[0], self.val[1], self.val[2], 0.1, 50).tolist()
        self.super_best_theta = compute_grad(self.val[0], self.val[1], self.val[2], 0.1, 1500)

    def test_prepare_values(self):
        # X_norm - first row
        self.assertAlmostEqual(self.x[0][0], 1)
        self.assertAlmostEqual(self.x[0][1], 0.13001, 5)
        self.assertAlmostEqual(self.x[0][2], -0.22368, 5)

        # X_norm - last row
        self.assertAlmostEqual(self.x[-1][0], 1)
        self.assertAlmostEqual(self.x[-1][1], -1.00370, 4)
        self.assertAlmostEqual(self.x[-1][2], -0.22368, 5)

        # y - first and last item
        self.assertAlmostEqual(self.y[0][0], 3.999e+005, 3)
        self.assertAlmostEqual(self.y[-1][0], 2.395e+005, 3)

        # mu
        self.assertAlmostEqual(self.mu[0], 2000.7, 1)
        self.assertAlmostEqual(self.mu[1], 3.1702, 4)

        # sigma
        self.assertAlmostEqual(self.sigma[0], 794.7, 1)
        self.assertAlmostEqual(self.sigma[1], 0.76098, 5)

    def test_grad(self):
        self.assertAlmostEqual(self.best_theta[0][0], 338658.249249, 6)
        self.assertAlmostEqual(self.best_theta[1][0], 104127.515597, 6)
        self.assertAlmostEqual(self.best_theta[2][0], -172.205334, 6)

    def test_cost(self):
        self.assertAlmostEqual(compute_cost(self.val[0], self.val[1], self.val[2]), 65591548106.457443, 6)
        self.assertAlmostEqual(compute_cost(self.val[0], np.array(self.best_theta), self.val[2]), 2062961418.085968, 6)

    def test_normal_eq_and_gradient_descent(self):
        self.assertAlmostEqual(compute_cost(self.val[0], self.super_best_theta, self.val[2]),
                               compute_cost(self.val[0], normal_equation(self.val[0], self.val[2]), self.val[2]), 5)


if __name__ == '__main__':
    unittest.main()
