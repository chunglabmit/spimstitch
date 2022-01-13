import unittest
import numpy as np
import spimstitch.utils

class TestWeightedMedian(unittest.TestCase):

    def do_case(self, data, weights, expected_result):
        result = spimstitch.utils.weighted_median(data, weights)
        self.assertAlmostEqual(result, expected_result, delta=.1)

    def test_nan(self):
        result = spimstitch.utils.weighted_median([], [])
        self.assertTrue(np.isnan(result))

    def test_one(self):
        self.do_case([15], [1], 15)

    def test_low(self):
        self.do_case([15, 16, 17], [3, 1, 1], 15)

    def test_high(self):
        self.do_case([15, 16, 17], [1, 1, 3], 17)

    def test_middle(self):
        self.do_case([15, 16, 17], [1.5, 1, 1], 15 + 1/3 )


if __name__ == '__main__':
    unittest.main()
