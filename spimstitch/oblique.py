from blockfs.directory import Directory
import functools
import itertools
import numpy as np
import os
import tifffile
import typing
from .stack import make_stack, SpimStack, StackFrame
from .shared_memory import  SharedMemory
from .pipeline import Resource, Dependent, Pipeline

DIRECTORY = None

def get_blockfs_dims(stack:SpimStack) -> typing.Tuple[int,int,int, np.dtype]:
    """
    Determine the dimensions of the blockfs directory that will be needed
    to write the given stack

    :param stack: the stack to be written
    :return: a 4-tuple of z extent, y extent and x extent and image dtype
    """
    sf = StackFrame(stack.paths[0], stack.x0, stack.y0, stack.z0)
    x_extentish = sf.x1 - sf.x0
    y_extent = sf.y1 - sf.y0
    z_extentish = len(stack.paths)
    x_extent = x_extentish + z_extentish
    z_extent = x_extentish
    return z_extent, y_extent, x_extent, sf.img.dtype

class PlaneR:

    def __init__(self, z, path, shape, dtype):
        self.z = z
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.memory = None

    def prepare(self):
        if self.memory is None:
            self.memory = SharedMemory(self.shape, self.dtype)

    def read(self):
        with self.memory.txn() as m:
            m[:] = tifffile.imread(self.path)


class BlockD:

    def __init__(self,
                 planes:typing.Sequence[PlaneR],
                 x0:int, x1:int, y0:int, y1:int, z0:int, z1:int):
        self.planes = planes
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def execute(self):
        block = np.zeros((self.z1 - self.z0,
                          self.y1 - self.y0,
                          self.x1 - self.x0), DIRECTORY.dtype)
        for plane in self.planes:
            px0 = plane.z + self.z0
            px1 = plane.z + self.z1
            x0a = max(self.x0, px0)
            x1a = min(self.x1, px1)
            y0a = max(self.y0, 0)
            y1a = min(self.y1, plane.shape[0])

            srcy, srcx = np.mgrid[y0a:y1a, x0a - plane.z:x1a - plane.z]
            desty, destx = np.mgrid[y0a - self.y0:y1a - self.y0,
                                    x0a - self.x0:x1a - self.x0]
            destz = srcx - self.z0
            with plane.memory.txn() as m:
                block[destz.flatten(), desty.flatten(), destx.flatten()] = \
                    m[srcy.flatten(), srcx.flatten()]
        DIRECTORY.write_block(block, self.x0, self.y0, self.z0)



def make_resources(stack:SpimStack)\
        -> typing.Sequence[typing.Tuple[Resource, PlaneR]]:
    result = []
    img = tifffile.imread(stack.paths[0])
    shape = img.shape
    dtype = img.dtype
    for z, path in enumerate(stack.paths):
        planer = PlaneR(z, path, shape, dtype)
        resource = Resource(planer.read, "plane %d" % z, planer.prepare)
        result.append((resource, planer))
    return result


def spim_to_blockfs(stack:SpimStack, directory:Directory, n_workers):
    global DIRECTORY
    DIRECTORY = directory
    dependents = make_s2b_dependents(stack, directory)
    pipeline = Pipeline(dependents)
    pipeline.run(n_workers)


def make_s2b_dependents(stack:SpimStack, directory:Directory):
    resources_and_planers = make_resources(stack)
    resources, planers = [[_[idx] for _ in resources_and_planers]
                          for idx in (0, 1)]

    x0r = range(0, directory.x_extent, directory.x_block_size)
    x1r = [min(_+directory.x_block_size, directory.x_extent) for _ in x0r]
    y0r = range(0, directory.y_extent, directory.y_block_size)
    y1r = [min(_+directory.y_block_size, directory.y_extent) for _ in y0r]
    z0r = range(0, directory.z_extent, directory.z_block_size)
    z1r = [min(_+directory.z_block_size, directory.z_extent) for _ in z0r]
    dependents = []
    for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
        zip(x0r, x1r), zip(y0r, y1r), zip(z0r, z1r)):
        z0idx = max(stack.z0, x0 - z1 + 1)
        z1idx = min(stack.z1, x1 - z0)
        if z1idx <= z0idx:
            continue
        blockd = BlockD(planers[z0idx:z1idx],
                        x0, x1, y0, y1, z0, z1)
        dependent = Dependent(
            prerequisites=resources[z0idx:z1idx],
            fn=blockd.execute,
            name="block %d:%d %d:%d %d:%d" % (x0, x1, y0, y1, z0, z1))
        dependents.append(dependent)
    return dependents