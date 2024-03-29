import numpy as np
import zarr
from precomputed_tif.ngff_stack import NGFFStack


class NGFFDirectory:
    """A duck-type of the blockfs directory"""
    def __init__(self, stack:NGFFStack):
        """
        Connect to an existing NGFF stack

        :param stack: NGFF stack with valid zgroup (e.g. stack.create())
        """
        self.stack = stack

    def create(self):
        self.array = self.stack.create_dataset(1)

    @property
    def shape(self):
        return (self.stack.z_extent, self.stack.y_extent, self.stack.x_extent)

    @property
    def x_extent(self):
        return self.stack.x_extent

    @property
    def y_extent(self):
        return self.stack.y_extent

    @property
    def z_extent(self):
        return self.stack.z_extent

    @property
    def x_block_size(self):
        return self.stack.cx()

    @property
    def y_block_size(self):
        return self.stack.cy()

    @property
    def z_block_size(self):
        return self.stack.cz()

    @property
    def dtype(self):
        return self.stack.dtype

    def get_block_size(self, x, y, z):
        return (min(self.stack.z_extent - z, self.stack.cz()),
                min(self.stack.y_extent - y, self.stack.cy()),
                min(self.stack.x_extent - x, self.stack.cx()))

    def write_block(self, block:np.ndarray, x0:int, y0:int, z0:int):
        zs, ys, xs = self.get_block_size(x0, y0, z0)
        z1, y1, x1 = z0 + zs, y0 + ys, x0 + xs
        self.array[0, 0, z0:z1, y0:y1, x0:x1] = block

    def close(self):
        pass


class NGFFReadOnlyDirectory:
    """
    A duck type of blockfs Directory, except geared for read-only access.
    """

    def __init__(self, path, level=1):
        self.store = zarr.NestedDirectoryStore(path)
        self.zgroup = zarr.group(self.store, overwrite=False)
        ngff_level = int(np.round(np.log(level) / np.log(2)))
        self.array:zarr.Array = self.zgroup[str(ngff_level)]
        self.array.read_only = True

    @property
    def shape(self):
        return self.array.shape[2:]

    @property
    def x_extent(self):
        return self.shape[2]

    @property
    def y_extent(self):
        return self.shape[1]

    @property
    def z_extent(self):
        return self.shape[0]

    @property
    def x_block_size(self):
        return self.array.chunks[4]

    @property
    def y_block_size(self):
        return self.array.chunks[3]

    @property
    def z_block_size(self):
        return self.array.chunks[2]

    @property
    def dtype(self):
        return self.array.dtype

    def get_block_size(self, x, y, z):
        return (min(self.z_extent - z, self.z_block_size),
                min(self.y_extent - y, self.y_block_size),
                min(self.x_extent - x, self.x_block_size))

    def read_block(self, x0, y0, z0):
        zs, ys, xs = self.get_block_size(x0, y0, z0)
        x1 = x0 + xs
        y1 = y0 + ys
        z1 = z0 + zs
        return self.array[0, 0, z0:z1, y0:y1, x0:x1]