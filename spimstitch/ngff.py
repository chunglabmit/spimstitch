import numpy as np
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

