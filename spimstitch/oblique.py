from blockfs.directory import Directory
import itertools
import multiprocessing
import numpy as np
import tifffile
import typing

from scipy.ndimage import map_coordinates

from .stack import SpimStack, StackFrame
from mp_shared_memory import SharedMemory
from .pipeline import Resource, Dependent, Pipeline

DIRECTORY:Directory = None
L2_DIRECTORY:Directory = None


def oblique2um(x:typing.Union[int, float, np.ndarray],
               y:typing.Union[int, float, np.ndarray],
               frame:typing.Union[int, float, np.ndarray],
               voxel_size:float,
               x_step_size:float) -> \
        typing.Tuple[typing.Union[float, np.ndarray],
                     typing.Union[float, np.ndarray],
                     typing.Union[float, np.ndarray]]:
    """Convert coordinate positions from DCIMG to microns

    :param x: X coordinate or coordinates on a DCIMG plane
    :param y: Y coordinate or coordinates on a DCIMG plane
    :param frame: the frame number of the plane
    :param voxel_size: the size of a plane voxel in microns
    :param x_step_size: the # of microns between X steps
    :returns: The coordinates in microns as z, y, x
    """
    x_out = frame * x_step_size + x * voxel_size / np.sqrt(2)
    y_out = y * voxel_size / np.sqrt(2)
    z_out = x * voxel_size / np.sqrt(2)
    return z_out, y_out, x_out


def um2oblique(x:typing.Union[int, float, np.ndarray],
               y:typing.Union[int, float, np.ndarray],
               z:typing.Union[int, float, np.ndarray],
               voxel_size:float,
               x_step_size:float) -> typing.Tuple[
    typing.Union[float, np.ndarray],
    typing.Union[float, np.ndarray],
    typing.Union[float, np.ndarray]]:
    """Convert coordinate positions from microns to DCIMG-relative

    :param x: X coordinate or coordinates in microns
    :param y: Y coordinate or coordinates in microns
    :param z: Z coordinate or coordinates in microns
    :param voxel_size: Size of a voxel in the DCIMG plane in microns
    :param x_step_size: distance between planes in microns
    :returns: a three-tuple of frame , y and x
    """
    frame = x / x_step_size - z / x_step_size
    x_out = z * np.sqrt(2) / voxel_size
    y_out = y / voxel_size
    return frame, y_out, x_out


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
                 ys:int,
                 x_step_size:float,
                 voxel_size:float):
        """

        :param planes: The PlaneR plane placeholders for the block
        :param x0: The X start of the block in voxels
        :param x1: The X end of the block
        :param y0: The Y start of the block
        :param y1: The Y end of the block
        :param z0: The Z start of the block
        :param z1: The Z end of the block
        :param ys: the size of a blockfs block in the Y direction
        :param x_step_size: the step size in microns
        :param voxel_size: the voxel size in microns
        """
        self.planes = planes
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.ys = ys
        self.x_step_size = x_step_size
        self.voxel_size = voxel_size

    def execute(self):
        block = np.zeros((self.z1 - self.z0,
                          self.y1 - self.y0,
                          self.x1 - self.x0), np.float32)
        if np.abs(self.x_step_size - self.voxel_size / np.sqrt(2)) < .01:
            # Isotropic x / z case. Get exact pixel locations. Code is faster.
            for plane in self.planes:
                x0a = self.x0
                x1a = self.x1
                y0a = max(self.y0, 0)
                y1a = min(self.y1, plane.shape[0])

                srcy, srcx = np.mgrid[y0a:y1a, x0a - plane.z:x1a - plane.z]
                desty, destx = np.mgrid[y0a - self.y0:y1a - self.y0,
                               x0a - self.x0:x1a - self.x0]
                destz = srcx - self.z0
                mask = (destz >= 0) & (destz < block.shape[0]) &\
                       (srcx >= 0) & (srcx < plane.shape[1])
                with plane.memory.txn() as m:
                    block[destz[mask], desty[mask], destx[mask]] = \
                        m[srcy[mask], srcx[mask]]
        else:
            # Anisotropic x/z case. Interpolate the voxels.
            zvox, yvox, xvox = np.mgrid[self.z0:self.z1,
                                     self.y0:self.y1,
                                     self.x0:self.x1]
            zum = self.voxel_size * zvox / np.sqrt(2)
            yum = self.voxel_size * yvox
            xum = xvox * self.x_step_size
            frame, yo, xo = um2oblique(xum, yum, zum,
                                       voxel_size=self.voxel_size,
                                       x_step_size=self.x_step_size)
            frame_lo = np.floor(frame).astype(np.int32)
            frame_hi = frame_lo + 1
            frame_lo_frac = 1 - frame + frame_lo
            frame_hi_frac = 1 - frame_hi + frame
            for plane in self.planes:
                mask_lo = plane.z == frame_lo
                mask_hi = (plane.z == frame_hi) & (frame_hi > 0)
                with plane.memory.txn() as m:
                    if np.any(mask_lo) and np.any(frame_lo_frac[mask_lo] > 0):
                        block[zvox[mask_lo] - self.z0,
                              yvox[mask_lo] - self.y0,
                              xvox[mask_lo] - self.x0] += \
                            map_coordinates(m, (yo[mask_lo], xo[mask_lo]),
                                            order=1) * \
                            frame_lo_frac[mask_lo]
                    if np.any(mask_hi) and np.any(frame_hi_frac[mask_hi] > 0):
                        block[zvox[mask_hi] - self.z0,
                              yvox[mask_hi] - self.y0,
                              xvox[mask_hi] - self.x0] += \
                            map_coordinates(m, (yo[mask_hi], xo[mask_hi]),
                                            order=1) * \
                            frame_hi_frac[mask_hi]
        for x0a, y0a, z0a in itertools.product(
            range(self.x0, self.x1, DIRECTORY.x_block_size),
            range(self.y0, self.y1, DIRECTORY.y_block_size),
            range(self.z0, self.z1, DIRECTORY.z_block_size)):
            x1a = min(x0a + DIRECTORY.x_block_size, self.x1)
            y1a = min(y0a + DIRECTORY.y_block_size, self.y1)
            z1a = min(z0a + DIRECTORY.z_block_size, self.z1)
            DIRECTORY.write_block(
                block[z0a - self.z0:z1a - self.z0,
                      y0a - self.y0:y1a - self.y0,
                      x0a - self.x0:x1a - self.x0].astype(np.uint16),
                                  x0a, y0a, z0a)
        if L2_DIRECTORY is None:
            return
        block_e = block[:block.shape[0] & ~ 1,
                        :block.shape[1] & ~ 1,
                        :block.shape[2] & ~ 1]
        all_decimated = [block_e[x::2, y::2, z::2].astype(np.uint32)
                         for x, y, z in
                         itertools.product((0, 1), (0, 1), (0, 1))]
        block_d = (sum(all_decimated[1:], all_decimated[0]) // 8).\
            astype(np.uint16)
        x0_l2 = self.x0 // 2
        y0_l2 = self.y0 // 2
        z0_l2 = self.z0 // 2
        x1_l2 = self.x1 // 2
        y1_l2 = self.y1 // 2
        z1_l2 = self.z1 // 2
        for x0a, y0a, z0a in itertools.product(
            range(x0_l2, x1_l2, L2_DIRECTORY.x_block_size),
            range(y0_l2, y1_l2, L2_DIRECTORY.y_block_size),
            range(z0_l2, z1_l2, L2_DIRECTORY.z_block_size)):
            x1a = min(x0a + L2_DIRECTORY.x_block_size, x1_l2)
            y1a = min(y0a + L2_DIRECTORY.y_block_size, y1_l2)
            z1a = min(z0a + L2_DIRECTORY.z_block_size, z1_l2)
            try:
                L2_DIRECTORY.write_block(
                    block_d[z0a - z0_l2:z1a - z0_l2,
                           y0a - y0_l2:y1a - y0_l2,
                           x0a - x0_l2:x1a - x0_l2].astype(np.uint16),
                    x0a, y0a, z0a)
            except ValueError:
                print(f"x0={self.x0} y0={self.y0} z0={self.z0}")
                print(f"x1={self.x1} y1={self.y1} z1={self.z1}")
                print(f"x0a={x0a} y0a={y0a} z0a={z0a}")
                print(f"x1a={x1a} y1a={y1a} z1a={z1a}")
                print(f"L2_DIRECTORY.shape={L2_DIRECTORY.shape}")
                raise


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
                    voxel_size:float,
                    x_step_size:float,
                    read_fn:READ_FUNCTION_T=tifffile.imread,
                    l2_directory:Directory=None):
    global DIRECTORY, L2_DIRECTORY
    DIRECTORY = directory
    L2_DIRECTORY = l2_directory
    dependents = make_s2b_dependents(stack, directory,
                                     l2_directory,
                                     voxel_size, x_step_size, read_fn)
    pipeline = Pipeline(dependents)
    del dependents
    with multiprocessing.Pool(n_workers) as pool:
        pipeline.run(pool)
        directory.close()
        DIRECTORY = None


def make_s2b_dependents(stack:SpimStack,
                        directory:Directory,
                        l2_directory:Directory,
                        voxel_size:float,
                        x_step_size:float,
                        read_fn:READ_FUNCTION_T=tifffile.imread):
    resources_and_planers = make_resources(stack,
                                           read_fn)
    resources, planers = [[_[idx] for _ in resources_and_planers]
                          for idx in (0, 1)]

    x0r = range(0, directory.x_extent, directory.x_block_size * 2)
    x1r = [min(_+directory.x_block_size*2, directory.x_extent) for _ in x0r]
    z0r = range(0, directory.z_extent, directory.z_block_size)
    z1r = [min(_+directory.z_block_size*2, directory.z_extent) for _ in z0r]
    dependents = []
    for (x0, x1), (z0, z1) in itertools.product(
        zip(x0r, x1r), zip(z0r, z1r)):
        x0um = x0 * x_step_size
        x1um = x1 * x_step_size
        z0um = z0 * voxel_size / np.sqrt(2)
        z1um = z1 * voxel_size / np.sqrt(2)
        # Get the frame # of all 4 corners for minmax
        frame, _, _ = um2oblique(np.array([x0um, x1um, x0um, x1um]),
                                 np.array([0, 0, 0, 0]),
                                 np.array([z0um, z0um, z1um, z1um],),
                                 voxel_size=voxel_size,
                                 x_step_size=x_step_size)
        z0idx = max(stack.z0, int(np.min(frame)))
        z1idx = min(stack.z1, int(np.max(np.ceil(frame))))
        if z1idx <= z0idx:
            continue
        blockd = BlockD(planers[z0idx:z1idx],
                        x0, x1, 0, directory.y_extent, z0, z1,
                        directory.y_block_size,
                        voxel_size=voxel_size,
                        x_step_size=x_step_size)
        dependent = Dependent(
            prerequisites=resources[z0idx:z1idx],
            fn=blockd.execute,
            name="block %d:%d %d:%d" % (x0, x1, z0, z1))
        dependents.append(dependent)
    return dependents