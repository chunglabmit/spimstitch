from blockfs.directory import Directory
import itertools
import multiprocessing
import numpy as np
import tifffile
import typing
from .stack import SpimStack, StackFrame
from mp_shared_memory import SharedMemory
from .pipeline import Resource, Dependent, Pipeline

DIRECTORY = None


def get_blockfs_dims(stack:SpimStack,
                     x_extent=None,
                     y_extent=None) -> typing.Tuple[int,int,int, np.dtype]:
    """
    Determine the dimensions of the blockfs directory that will be needed
    to write the given stack

    :param stack: the stack to be written
    :return: a 4-tuple of z extent, y extent and x extent and image dtype
    """
    if x_extent is not None:
        x_extentish = x_extent
        dtype = np.uint16
    else:
        sf = StackFrame(stack.paths[0], stack.x0, stack.y0, stack.z0)
        x_extentish = sf.x1 - sf.x0
        y_extent = sf.y1 - sf.y0
        dtype = sf.img.dtype
    z_extentish = len(stack.paths)
    x_extent = x_extentish + z_extentish
    z_extent = x_extentish
    return z_extent, y_extent, x_extent, dtype


READ_FUNCTION_T = typing.Callable[[str], np.ndarray]
class PlaneR:

    def __init__(self, z:int, path:str, shape:typing.Sequence[int],
                 dtype:np.dtype, read_fn:READ_FUNCTION_T=tifffile.imread):
        self.z = z
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.memory = None
        self.read_fn = read_fn

    def prepare(self):
        if self.memory is None:
            self.memory = SharedMemory(self.shape, self.dtype)

    def read(self):
        with self.memory.txn() as m:
            m[:] = self.read_fn(self.path)


class BlockD:

    def __init__(self,
                 planes:typing.Sequence[PlaneR],
                 x0:int, x1:int, y0:int, y1:int, z0:int, z1:int,
                 ys:int):
        """

        :param planes: The PlaneR plane placeholders for the block
        :param x0: The X start of the block
        :param x1: The X end of the block
        :param y0: The Y start of the block
        :param y1: The Y end of the block
        :param z0: The Z start of the block
        :param z1: The Z end of the block
        :param ys: the size of a blockfs block in the Y direction
        """
        self.planes = planes
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.ys = ys

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
        for y0a in range(self.y0, self.y1, self.ys):
            y1a = min(y0a + self.ys, self.y1)
            DIRECTORY.write_block(block[:, y0a - self.y0:y1a - self.y0],
                                  self.x0, y0a, self.z0)


def make_resources(stack:SpimStack, read_fn:READ_FUNCTION_T=tifffile.imread)\
        -> typing.Sequence[typing.Tuple[Resource, PlaneR]]:
    result = []
    img = read_fn(stack.paths[0])
    shape = img.shape
    dtype = img.dtype
    for z, path in enumerate(stack.paths):
        planer = PlaneR(z, path, shape, dtype, read_fn)
        resource = Resource(planer.read, "plane %d" % z, planer.prepare)
        result.append((resource, planer))
    return result


def spim_to_blockfs(stack:SpimStack, directory:Directory,
                    n_workers:int,
                    read_fn:READ_FUNCTION_T=tifffile.imread):
    global DIRECTORY
    DIRECTORY = directory
    dependents = make_s2b_dependents(stack, directory, read_fn)
    pipeline = Pipeline(dependents)
    with multiprocessing.Pool(n_workers) as pool:
        pipeline.run(pool)
        directory.close()
        DIRECTORY = None


def make_s2b_dependents(stack:SpimStack,
                        directory:Directory,
                        read_fn:READ_FUNCTION_T=tifffile.imread):
    resources_and_planers = make_resources(stack, read_fn)
    resources, planers = [[_[idx] for _ in resources_and_planers]
                          for idx in (0, 1)]

    x0r = range(0, directory.x_extent, directory.x_block_size)
    x1r = [min(_+directory.x_block_size, directory.x_extent) for _ in x0r]
    z0r = range(0, directory.z_extent, directory.z_block_size)
    z1r = [min(_+directory.z_block_size, directory.z_extent) for _ in z0r]
    dependents = []
    for (x0, x1), (z0, z1) in itertools.product(
        zip(x0r, x1r), zip(z0r, z1r)):
        z0idx = max(stack.z0, x0 - z1 + 1)
        z1idx = min(stack.z1, x1 - z0)
        if z1idx <= z0idx:
            continue
        blockd = BlockD(planers[z0idx:z1idx],
                        x0, x1, 0, directory.y_extent, z0, z1,
                        directory.y_block_size)
        dependent = Dependent(
            prerequisites=resources[z0idx:z1idx],
            fn=blockd.execute,
            name="block %d:%d %d:%d" % (x0, x1, z0, z1))
        dependents.append(dependent)
    return dependents