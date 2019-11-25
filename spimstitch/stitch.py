from blockfs.directory import Directory
import functools
import numpy as np
from .shared_memory import SharedMemory
from .stack import make_stack
from .pipeline import Resource, Dependent, Pipeline
import typing
import tifffile


class Plane:

    def __init__(self,
                 path:str,
                 x0:int,
                 y0:int,
                 z:int,
                 shape:typing.Sequence[int],
                 dtype:np.dtype):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.memory = None
        self.x0 = x0
        self.x1 = x0 + shape[2]
        self.y0 = y0
        self.y1 = y0 + shape[1]
        self.z = z

    def prepare(self):
        self.memory = SharedMemory(self.shape, self.dtype)

    def read(self):
        self.memory[:] = tifffile.imread(self.path)

    def name(self):
        return "plane x=%d-%d y=%d-%d z=%d" % (
            self.x0, self.x1, self.y0, self.y1, self.z)

    def intersection(self, other) -> typing.Tuple[int, int, int, int]:
        """

        Return the intersection of self with another plane. Assumes that both
        are at the same z

        :param other: the other plane
        :return: a tuple of x0, x1, y0 which is the intersection
        """
        x0 = max(self.x0, other.x0)
        x1 = min(self.x1, other.x1)
        y0 = max(self.y0, other.y0)
        y1 = min(self.y1, other.y1)
        return x0, x1, y0, y1

class Planes:
    def __init__(self, planes:typing.Sequence[Plane]):
        self.planes_by_z = {}
        for plane in planes:
            if plane.z not in self.planes_by_z:
                self.planes_by_z[plane.z] = []
            self.planes_by_z[plane.z].append(
                (plane,
                 Resource(plane.read, plane.name(), plane.prepare)))

    def find_overlapping_planes(self, x0:int, x1:int, y0:int, y1:int, z:int):
        if z not in self.planes_by_z:
            return []
        return [
            (plane, resource) for plane, resource in self.planes_by_z
            if plane.x0 < x1 and
               plane.x1 > x0 and
               plane.y0 < y1 and
               plane.y1 < y0
        ]


def make_plane_dependent(
        planes:Planes,
        fn:typing.Callable[[], typing.Sequence[typing.Sequence[Plane]]],
        x0:int,
        x1:int,
        y0:int,
        y1:int,
        z0:int,
        z1:int) -> Dependent:
    """
    Make a Dependant that calls the given function when all of its planes
    are ready
    :param planes: a Planes that can be used to find planes for the block
    :param fn: the function to run. The argument to the callable is a list
    with one element per Z between z0 and z1 and the element itself is a list
    of all planes that make up that Z
    :param x0: x minimum of block
    :param x1: x maximum of block
    :param y0: y minimum of block
    :param y1: y maximum of block
    :param z0: z minimum of block
    :param z1: z maximum of block
    :return: a dependant task that does something to the block
    """
    resources = []
    planes_seq = []
    for z in range(z0, z1):
        planes_per_z = []
        for resource, plane in \
                planes.find_overlapping_planes(x0, x1, y0, y1, z):
            resources.append(resource)
            planes_per_z.append(plane)
        planes_seq.append(planes_per_z)
    function = functools.partial(fn, planes_seq)
    return Dependent(resources, function,
                     "block x=%d-%d y=%d-%d z=%d-%d" % (x0, x1, y0, y1, z0, z1))


def write_block(directory:Directory,
                x0:int,
                y0:int,
                z0:int,
                planes:typing.Sequence[typing.Sequence[Plane]]):
    """
    This is the workhorse that writes a Blockfs block given a pile of planes

    :param directory: the blockfs directory - the writers should be started
    on it.
    :param x0: the x start of the block
    :param y0: the y start of the block
    :param z0: the z start of the block
    :param planes: the planes that go into the block. These should be read
    by the time we get here.
    """
    zs, ys, xs = directory.get_block_size(x0, y0, z0)
    x1 = x0 + xs
    y1 = y0 + ys
    z1 = z0 + zs
    dest = np.zeros((zs, ys, xs), directory.dtype)
    for zidx, planes in enumerate(planes):
        z = z0 + zidx
        if len(planes) == 0:
            continue # No data for this z
        elif len(planes) == 1:
            plane = planes[0]
            x0a = max(x0, plane.x0)
            x1a = min(x1, plane.x1)
            y0a = max(y0, plane.y0)
            y1a = min(y1, plane.y1)
            with plane.memory.txn() as m:
                dest[zidx, y0a - y0:y1a - y0, x0a - x0:x1a - x0] = \
                    m[y0a - plane.y0:y1a - plane.y0,
                      x0a - plane.x0:x1a - plane.x0]
        else:
            # here we have to blend boundaries.
            # This should be so rare that we can take our time with
            # the calculation.
            for i, plane in enumerate(planes):
                multiplier = np.ones(plane.shape)
                for j, other_plane in enumerate(planes):
                    if i == j:
                        continue
                    x0i, x1i, y0i, y1i = plane.intersection(other_plane)
                    if x0i >= x1i or y0i >= y1i:
                        continue # no overlap
                    d = np.ones((y1i - y0i, x1i-x0i)) * max(x1i-x0i, y1i-y0i)
                    od = d.copy()
                    s = d.copy()
                    if plane.x0 == x0i and other_plane.x1 == x1i:
                        # intersection might be on the right
                        d[:] = s + np.arange(0, x1i - x0a).reshape(1, -1)
                        od[:] = s + np.arange(0, x1i - x0a)[::-1].reshape(1, -1)
                    elif plane.x1 == x1i and other_plane.x0 == x0i:
                        od[:] = s + np.arange(0, x1i - x0a).reshape(1, -1)
                        d[:] = s + np.arange(0, x1i - x0a)[::-1].reshape(1, -1)
                    if plane.y0 == y0i and other_plane.y1 == y1i:
                        d = np.minimum(
                            d, s + np.arange(0, y1i - y0i).reshape(-1, 1))
                        od = np.minimum(
                            od,
                            s + np.arange(0, y1i - y0i)[::-1].reshape(-1, 1))
                    elif plane.y1 == y1i and other_plane.y0 == y0i:
                        od = np.minimum(
                            d, s + np.arange(0, y1i - y0i).reshape(-1, 1))
                        d = np.minimum(
                            od,
                            s + np.arange(0, y1i - y0i)[::-1].reshape(-1, 1))
                    angle = np.arctan(d, od)
                    blending = np.sin(angle) ** 2
                    multiplier[y0i - y0:y1i - y0, x0i - x0:x1i - x0] *= blending
                with plane.memory.txn() as m:
                    img = (m * multiplier).astype(Directory.dtype)
                    dest[zidx, y0a - y0:y1a - y0, x0a - x0:x1a - x0] = \
                        img[y0a - plane.y0:y1a - plane.y0,
                            x0a - plane.x0:x1a - plane.x0]
    directory.write_block(dest, x0, y0, z0)


def make_plane_dependents(directory:Directory,
                          planes:Planes):
    dependents = []
    for x0 in range(0, directory.x_extent, directory.x_block_size):
        x1 = min(x0 + directory.x_block_size, directory.x_extent)
        for y0 in range(0, directory.y_extent, directory.y_block_size):
            y1 = min(y0 + directory.y_block_size, directory.y_extent)
            for z0 in range(0, directory.z_extent, directory.z_block_size):
                z1 = min(z0 + directory.z_block_size, directory.z_extent)
                fn = functools.partial(write_block,
                                       directory=directory,
                                       x0=x0, y0=y0, z0=z0)
                dependent = make_plane_dependent(
                    planes, fn, x0, x1, y0, y1, z0, z1)
                dependents.append(dependent)
    return dependents


def make_planes(path, ext, shape, dtype):
    stacks = make_stack(path, ext, shape)
    planes = []
    for (x0, y0), stack in stacks.items():
        for idx, path in enumerate(stack.paths):
            x = x0 + idx
            z = idx
            plane = Plane(path, x, y0, z, shape, dtype)
            planes.append(plane)
    return Plane(planes)