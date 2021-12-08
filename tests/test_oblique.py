import contextlib
import logging
import numpy as np
import os
import shutil
import tempfile
import tifffile
import typing
import unittest
from spimstitch.oblique import spim_to_blockfs
from spimstitch.stack import SpimStack
from blockfs.directory import Directory
logging.basicConfig(level=logging.INFO)

@contextlib.contextmanager
def make_case(volume_shape:typing.Tuple[int, int, int],
              block_shape:typing.Tuple[int, int, int])\
        ->typing.Tuple[typing.Sequence[str], Directory, np.ndarray]:

    volume = np.random.RandomState(np.prod(volume_shape))\
        .randint(0, 65535,  volume_shape).astype(np.uint16)
    tempdir = tempfile.mkdtemp()
    directory = Directory(volume_shape[2] + volume_shape[0],
                          volume_shape[1],
                          volume_shape[2],
                          np.uint16,
                          os.path.join(tempdir, "test.blockfs"),
                          n_filenames=1,
                          x_block_size=block_shape[2],
                          y_block_size=block_shape[1],
                          z_block_size=block_shape[0])
    directory.create()
    directory.start_writer_processes()
    paths = []
    for i, plane in enumerate(volume):
        path = os.path.join(tempdir, "img_%04d.tiff" % i)
        paths.append(path)
        tifffile.imsave(path, plane)
    yield paths, directory, volume
    shutil.rmtree(tempdir, ignore_errors=True)


class TestOblique(unittest.TestCase):
    def test_pipeline(self):
        with make_case((128, 16, 16), (4, 4, 4)) as (paths, directory, volume):
            stack = SpimStack(paths, 0, 0, 16, 16, 0)
            spim_to_blockfs(stack, directory, 1,
                            voxel_size=3.625,
                            x_step_size=3.625 / (2.0 ** .5))
            directory.close()
            block = directory.read_block(0, 0, 0)
            np.testing.assert_array_equal(block[3, :, :3], 0)
            np.testing.assert_array_equal(block[0, :, 0], volume[0, :4, 0])
            np.testing.assert_array_equal(block[3, :, 3], volume[0, :4, 3])
            np.testing.assert_array_equal(block[0, :, 3], volume[3, :4, 0])
            r = np.random.RandomState(1234)
            for case in range(1000):
                xs = r.randint(0, 16)
                ys = r.randint(0, 16)
                zs = r.randint(0, 128)
                xd = xs + zs
                zd = xs
                xi = xd % 4
                yi = ys % 4
                zi = zd % 4
                block = directory.read_block(xd - xi, ys - yi, zd - zi)
                value = block[zi, yi, xi]
                expected = volume[zs, ys, xs]
                self.assertEqual(value, expected)

if __name__ == '__main__':
    unittest.main()
