import contextlib
import os
import unittest
import numpy as np
import shutil
import tempfile
from blockfs.directory import Directory

from spimstitch.stitch import StitchSrcVolume, get_output_size, run


class MockVolume:

    def __init__(self, path, x1, y1,  z1):
        x0 = y0 = z0 = 0
        trailing_oblique_start = x1 - z1 + z0
        z, y, x  = np.mgrid[z0:z1, y0:y1, x0:x1-z1]
        x = x+z
        volume = np.zeros((z1-z0, y1-y0, x1 - x0), np.uint16)
        r = np.random.RandomState(1234)
        volume[z.flatten(), y.flatten(), x.flatten()] = \
            r.randint(0, 65535, np.prod(z.shape))
        blockfs_path = os.path.join(path, "1_1_1", "precomputed.blockfs")
        os.makedirs(os.path.dirname(blockfs_path))
        directory = Directory(x1 - x0, y1 - y0, z1 - x0,
                              np.uint16, blockfs_path, n_filenames=1,
                              x_block_size=4,
                              y_block_size=4,
                              z_block_size=4)
        directory.create()
        directory.start_writer_processes()
        for xs in range(0, x1 - x0, 4):
            for ys in range(0, y1 - y0, 4):
                for zs in range(0, z1 - z0, 4):
                    directory.write_block(
                        volume[zs:zs + 4, ys:ys + 4, xs:xs + 4],
                        xs, ys, zs)
        directory.close()
        self.volume = volume
        self.path = path
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def rebase(self, x0, y0, z0=None):
        dx = x0 - self.x0
        self.x0 += dx
        self.x1 += dx
        dy = y0 - self.y0
        self.y0 += dy
        self.y1 += dy
        if z0:
            dz = z0 - self.z0
            self.z0 += dz
            self.z1 += dz


@contextlib.contextmanager
def make_case(vdescs):
    tempdir = tempfile.mkdtemp()
    volumes = []
    for xs, ys, zs, xum, yum in vdescs:
        path = os.path.join(tempdir, "%d_%d" % (xum, yum))
        volume = MockVolume(path, xs, ys, zs)
        volumes.append(volume)
    yield volumes
    shutil.rmtree(tempdir)


class TestStitch(unittest.TestCase):

    def test_make_volume(self):
        with make_case(((32, 16, 16, 100, 200),)) as volumes:
            mock_volume = volumes[0]
            volume = StitchSrcVolume(
                os.path.join(mock_volume.path, "1_1_1", "precomputed.blockfs"),
                1.8, 1.8, 0)
            self.assertAlmostEqual(volume.xum, 1.8 / np.sqrt(2), 3)
            self.assertAlmostEqual(volume.yum, 1.8, 3)
            self.assertAlmostEqual(volume.zum, 1.8 / np.sqrt(2), 3)
            self.assertEqual(volume.x0, 100)
            self.assertEqual(volume.y0, 200)

    def test_rebase_all(self):
        with make_case(((32, 16, 16, 100, 200),
                        (32, 16, 16, 150, 100),
                        (32, 16, 16, 50, 250))) as volumes:
            StitchSrcVolume.rebase_all(volumes)
            self.assertEqual(volumes[0].x0, 50)
            self.assertEqual(volumes[0].y0, 100)
            self.assertEqual(volumes[1].x0, 50)
            self.assertEqual(volumes[1].y0, 0)
            self.assertEqual(volumes[2].x0, 0)
            self.assertEqual(volumes[2].y0, 150)