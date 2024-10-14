import math
import unittest

import numpy as np


class TestLinear(unittest.TestCase):

    def setUp(self):
        self.linear_func = LinearFunc(np.array([2, 3]))

    def test_evaluate(self):
        self.assertEqual(self.linear_func(np.array([-5, 10])), 20)


class TestRastrigin(unittest.TestCase):

    def setUp(self):
        self.rastrigin_func = RastriginFunc()

    def test_local_optimum(self):
        self.assertTrue(math.isclose(self.rastrigin_func.local_optimum(np.array([1.2, 2.1]))[1], 5, abs_tol=1e-5))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
