# test the IO manager file
import io_manager
from matrix_operations import *
import unittest
import numpy as np


class TestIO(unittest.TestCase):
    def setUp(self):
        self.array = np.array([
            [1, 2, 3, 0],
            [1, 4, 2, 0],
            [1, 3, 21, 1],
            [213, 322, 231, 1.1]
        ], dtype=float)

    def test_load(self):
        self.assertTrue(matrix_equal(self.array, io_manager.load('test_data.txt')))


if __name__ == '__main__':
    unittest.main()
