import functools
import logging
import multiprocessing
import numpy as np
import unittest
from spimstitch.pipeline import Resource, Dependent, Pipeline
from spimstitch.shared_memory import SharedMemory

logging.basicConfig(level=logging.INFO)


class ComputePrimes:
    def __init__(self, n):
        self.shmem = None
        self.n = n

    def prepare(self):
        if self.shmem is None:
            self.shmem = SharedMemory((self.n,), bool)

    def set_2(self):
        with self.shmem.txn() as m:
            m[2] = True

    def set_range(self, i, j):
        with self.shmem.txn() as m:
            for idx in range(i, j):
                for d in range(2, int(np.ceil(np.sqrt(idx))) + 1):
                    if m[d] and idx % d == 0:
                        break
                else:
                    m[idx] = True

    def make_dependants(self, n=None):
        if n is None:
            n = self.n
        if n <= 2:
            return [Resource(self.set_2, "2", self.prepare)]
        else:
            base = int(np.ceil(np.sqrt(n)))
            prerequisites = self.make_dependants(base)
            dependants = []
            for i in range(base + 1, n, 25):
                j = min(n, i + 25)
                dependants.append(
                    Dependent(prerequisites,
                              functools.partial(self.set_range, i, j),
                              "Range %d-%d" % (i, j),
                              self.prepare)
                )
            return dependants


def null_function():
    pass


class TestPipeline(unittest.TestCase):
    def test_nothing(self):
        pipeline = Pipeline([])
        with multiprocessing.Pool(1) as pool:
            pipeline.run(pool)

    def test_simple(self):
        resource = Resource(null_function, "resource")
        dependant = Dependent([resource], null_function, "nothing")
        pipeline = Pipeline([dependant])
        with multiprocessing.Pool(1) as pool:
            pipeline.run(pool)

    def test_complex(self):
        cp = ComputePrimes(100)
        dependants = cp.make_dependants()
        pipeline = Pipeline(dependants)
        with multiprocessing.Pool(1) as pool:
            pipeline.run(pool)
        with cp.shmem.txn() as m:
            for i in range(2, 100):
                if i in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                         59, 61, 67, 71, 73, 79, 83, 89, 97):
                    self.assertTrue(m[i])
                else:
                    self.assertFalse(m[i])

if __name__ == '__main__':
    unittest.main()
