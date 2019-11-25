import contextlib
import os
import numpy as np
import sys
if sys.platform.startswith("linux"):
    is_linux = True
    import tempfile
else:
    is_linux = False
    import mmap


class SharedMemory:
    """A class to share memory between processes

    Instantiate this class in the parent process and use in all processes.

    For all but Linux, we use the mmap module to get a buffer for Numpy
    to access through numpy.frombuffer. But in Linux, we use /dev/shm which
    has no file backing it and does not need to deal with maintaining a
    consistent view of itself on a disk.

    Typical use:

    shm = SharedMemory((100, 100, 100), np.float32)

    def do_something():

        with shm.txn() as a:

            a[...] = ...

    with multiprocessing.Pool() as pool:

        pool.apply_async(do_something, args)

    """

    if is_linux:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            self.tempfile = tempfile.NamedTemporaryFile(
                prefix="proc_%d_" % os.getpid(),
                suffix=".shm",
                dir="/dev/shm",
                delete=True)
            self.pathname = self.tempfile.name
            self.shape = shape
            self.dtype = np.dtype(dtype)

        @contextlib.contextmanager
        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.memmap(self.pathname,
                               shape=self.shape,
                               dtype=self.dtype)
            yield memory
            del memory

        def __getstate__(self):
            return self.pathname, self.shape, self.dtype

        def __setstate__(self, args):
            self.pathname, self.shape, self.dtype = args

    else:
        def __init__(self, shape, dtype):
            """Initializer

            :param shape: the shape of the array

            :param dtype: the data type of the array
            """
            length = np.prod(shape) * dtype.itemsize
            self.mmap = mmap.mmap(-1, length)
            self.shape = shape
            self.dtype = dtype

        @contextlib.contextmanager
        def txn(self):
            """ A contextual wrapper of the shared memory

            :return: a view of the shared memory which has the shape and
            dtype given at construction
            """
            memory = np.frombuffer(self.mmap, self.shape, self.dtype)
            yield memory
            del memory
