import os
import h5py
from xml.dom.minidom import parse

import numpy as np


def parse_terastitcher(path: str, level:int = 1):
    from .stitch import StitchSrcVolume

    class ImarisStitchSrcVolume(StitchSrcVolume):
        def read_block(
                self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) \
                -> np.ndarray:
            x0r = self.x_relative(x0)
            y0r = self.y_relative(y0)
            z0r = self.z_relative(z0)
            x1r = self.x_relative(x1)
            y1r = self.y_relative(y1)
            z1r = self.z_relative(z1)
            block = np.zeros((z1-z0, y1-y0, x1-x0), self.directory.dtype)
            x0a, y0a, z0a = [max(r, 0) for r in (x0r, y0r, z0r)]
            z1a, y1a, x1a = \
                [min(r, lim) for r, lim in zip((z1r, y1r, x1r),
                                               self.directory.shape)]
            chunk = self.directory[z0a:z1a, y0a:y1a, x0a:x1a]
            block[z0a - z0r:z1a - z0r,
                  y0a - y0r:y1a - y0r,
                  x0a - x0r:x1a - x0r] = chunk
            return block

    dom = parse(path)
    root = dom.getElementsByTagName("TeraStitcher")[0]
    stacks_dir = root.getElementsByTagName("stacks_dir")[0]
    ims_dir = stacks_dir.attributes["value"].nodeValue
    voxel_dims = root.getElementsByTagName("voxel_dims")[0]
    voxel_size = dict([(name, float(voxel_dims.attributes[key].nodeValue))
                       for name, key in (("x", "H"), ("y", "V"), ("z", "D"))])
    stacks = root.getElementsByTagName("STACKS")[0]
    volumes = {}
    for stack in stacks.getElementsByTagName("Stack"):
        x = float(stack.attributes["ABS_H"].nodeValue) * voxel_size["x"]
        y = float(stack.attributes["ABS_V"].nodeValue) * voxel_size["y"]
        filename = stack.attributes["IMG_REGEX"].nodeValue
        path = os.path.join(ims_dir, filename)
        volumes[x, y, 0] = ImarisStitchSrcVolume(
            path,
            x_step_size=abs(voxel_size["x"]),
            yum=abs(voxel_size["y"]),
            zum=abs(voxel_size["z"]),
            x0=x,
            y0=y,
            z0=0,
            is_oblique=False,
            is_ims=True,
            level=level
        )
    return volumes


class ImarisReadOnlyDirectory:
    """
    A duck type of blockfs Directory, except geared for read-only access.
    """

    def __init__(self, path, level):
        self.path = path
        self.initialized = False
        self.current_channel = 0
        self.level = level

    def check_initialize(self):
        """
        The initialization is delayed so that it happens in each
        multiprocessing thread. This gives HDF5 a chance to make a
        process-specific file handle.
        """
        if not self.initialized:
            imaris_level = int(np.round(np.log2(self.level)))
            resolution_level = f"ResolutionLevel {imaris_level}"
            self.h5file = h5py.File(self.path, "r")
            r0t0 = self.h5file["DataSet"][resolution_level]["TimePoint 0"]
            self.channels = [r0t0[k]["Data"] for k in r0t0
                             if k.startswith("Channel")]
            self.initialized = True

    @property
    def shape(self):
        try:
            return self.__shape
        except AttributeError:
            self.cache_shape_and_dtype()
            return self.__shape

    def cache_shape_and_dtype(self):
        with h5py.File(self.path, "r") as fd:
            r0t0 = fd["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]
            ds = r0t0["Channel 0"]["Data"]
            self.__shape = ds.shape
            self.__dtype = ds.dtype

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
        self.check_initialize()
        return self.channels[self.current_channel].chunks[2]

    @property
    def y_block_size(self):
        self.check_initialize()
        return self.channels[self.current_channel].chunks[1]

    @property
    def z_block_size(self):
        self.check_initialize()
        return self.channels[self.current_channel].chunks[0]

    @property
    def dtype(self):
        try:
            return self.__dtype
        except AttributeError:
            self.cache_shape_and_dtype()
            return self.__dtype

    def __getitem__(self, slices):
        self.check_initialize()
        a = self.channels[self.current_channel]
        return a[slices]
