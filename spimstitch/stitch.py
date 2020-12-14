# -*- coding: utf-8 -*-

from blockfs.directory import Directory
import itertools
import multiprocessing
import numpy as np
import os
import tqdm
import typing
import uuid

from scipy import ndimage


class StitchSrcVolume:

    def __init__(self, path:str, x_step_size:float, yum:float, z0:int,
                 is_oblique:bool=True):
        """

        :param path: Path to a blockfs directory file. The path contains
        position metadata for the volume - the path should be something like
        %d_%d/1_1_1/precomputed.blockfs
        :param x_step_size: stepper size in the X direction. We assume that the
        X pixel size is sqrt(2) * x_step_size, so the Z pixel size is the
        same as the stepper size.
        :param yum: Y pixel size of pixels in the original planes
        :param z0: z0 of the stack.
        :param is_oblique: True if oblique, False if doing non-oblique stitch
        """
        self.key = uuid.uuid4()
        self.path = path
        self.directory = Directory.open(path)
        self.xum = x_step_size
        self.zum = x_step_size
        self.yum = yum
        z_metadata_dir = os.path.dirname(os.path.dirname(path))
        metadata_dir = os.path.split(os.path.dirname(z_metadata_dir))[-1]
        self.x0, self.y0 = [int(_) / 10 for _ in metadata_dir.split("_")]
        self.z0 = z0
        self.is_oblique = is_oblique

    def rebase(self, x0:float, y0:float, z0:float=None)->type(None):
        """
        Once we find out where the top left corner is, we rebase all the
        stacks so that the first one starts at 0.

        :param x0: The absolute left of all of the stacks
        :param y0: The absolute top of all of the stacks
        :param z0: The absolute z-top of the stacks, if present
        """
        self.x0 -= x0
        self.y0 -= y0
        if z0 is not None:
            self.z0 -= z0

    @staticmethod
    def rebase_all(volumes:typing.Sequence["StitchSrcVolume"], z_too=False):
        """
        Rebase all the volumes so that the whole space starts at zero

        :param volumes: the volumes to be rebased
        """
        x0 = volumes[0].x0
        y0 = volumes[0].y0
        z0 = volumes[0].z0
        for volume in volumes[1:]:
            x0 = min(x0, volume.x0)
            y0 = min(y0, volume.y0)
            z0 = min(z0, volume.z0)
        for volume in volumes:
            if z_too:
                volume.rebase(x0, y0, z0)
            else:
                volume.rebase(x0, y0)

    @property
    def trailing_oblique_start(self):
        if self.is_oblique:
            return self.directory.x_extent - self.directory.z_extent
        else:
            return self.directory.x_extent

    def x_overlap(self, other:"StitchSrcVolume") -> int:
        """
        Compute the overlap between two stacks in the X direction in pixels.
        The "self" stack is the one to the left (xmin) and the "other" stack
        is the one to the right. The call assumes the two stacks have the
        same Z

        :param other: the stack to the right
        :return: the overlap in integer pixels. The location of the "other"
        stack's start is the end of this stack minus the overlap and the
        location of this stack in the "other" stack is the overlap.
        """
        # This is the stack length from the top left corner to the top right
        # corner or from the bottom left to the bottom right.
        #
        stack_length_um = \
               (self.directory.x_extent - self.directory.z_extent) * self.xum
        stack_end_um = self.x0 + stack_length_um
        overlap_length_um = stack_end_um - other.x0
        return overlap_length_um / self.xum

    def y_overlap(self, other:"StitchSrcVolume") -> int:
        """
        Compute the overlap between two stacks in the Y direction in pixels.
        The "self" stack is before the "other" stack.

        :param other: The stack "after" (at greater Y) "self"
        :return: the integer overlap between the stacks in pixels
        """
        stack_end_um = self.y0 + self.directory.y_extent * self.yum
        overlap_length_um = stack_end_um - other.y0
        return overlap_length_um / self.yum

    def does_overlap(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int)\
            -> bool:
        """
        Return True if the block volume overlaps this volume.

        :param x0: x start in pixels
        :param x1: x end in pixels
        :param y0: y start in pixels
        :param y1: y end in pixels
        :param z0: z start in pixels
        :param z1: z end in pixels
        :return: True if any of the volume overlaps
        """
        x0_relative = self.x_relative(x0)
        if x0_relative >= self.directory.x_extent:
            return False
        x1_relative = self.x_relative(x1)
        if x1_relative < 0:
            return False
        y0_relative = self.y_relative(y0)
        if y0_relative >= self.directory.y_extent:
            return False
        y1_relative = self.y_relative(y1)
        if y1_relative < 0:
            return False
        z0_relative = self.z_relative(z0)
        if z0_relative >= self.directory.z_extent:
            return False
        z1_relative = self.z_relative(z1)
        if z1_relative < 0:
            return False
        if self.is_oblique:
            #
            # The corner cases
            #
            if x1_relative <= z1_relative:
                return False # top corner of block is below the leading oblique
            # bottom corner of block is above the trailing oblique
            return x1_relative - self.trailing_oblique_start > z0_relative
        else:
            return True

    def find_overlap(self, other:"StitchSrcVolume") ->\
            typing.Tuple[typing.Tuple[int, int, int],
                         typing.Tuple[int, int, int]]:
        """
        Find the overlap of this volume with another

        :param other: the other volume
        :return: the overlap in global coordinates as two three-tuples,
        the first being the minimum corner (z, y, x) and the second
        the maximum.
        """
        x0i = max(self.x0_global, other.x0_global)
        x1i = min(self.x1_global, other.x1_global)
        y0i = max(self.y0_global, other.y0_global)
        y1i = min(self.y1_global, other.y1_global)
        z0i = max(self.z0_global, other.z0_global)
        z1i = min(self.z1_global, other.z1_global)
        return (z0i, y0i, x0i), (z1i, y1i, x1i)

    def is_inside(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int)\
            -> bool:
        """
        Return True if the block is wholly inside the volume

        :param x0: x start in pixels
        :param x1: x end in pixels
        :param y0: y start in pixels
        :param y1: y end in pixels
        :param z0: z start in pixels
        :param z1: z end in pixels
        :return: True if the block is wholly within the volume
        """
        x0r = self.x_relative(x0)
        y0r = self.y_relative(y0)
        z0r = self.z_relative(z0)
        if any([_ < 0 for _ in (x0r, y0r, z0r)]):
            return False
        if self.is_oblique:
            if x0r < z0r:
                return False
            if x0r + x1 - x1 - self.trailing_oblique_start > z0r + z1 - z0:
                return False
        if y0r + y1 - y0 > self.directory.y_extent:
            return False
        if z0r + z1 - z0 > self.directory.z_extent:
            return False
        return True

    def z_relative(self, z):
        return int(z - self.z0 / self.zum + .5)

    def y_relative(self, y):
        return int(y - self.y0 / self.yum + .5)

    def x_relative(self, x):
        return int(x - self.x0 / self.xum + .5)

    @property
    def x0_global(self):
        return int(self.x0 / self.xum + .5)

    @property
    def x1_global(self):
        return self.x0_global + self.directory.x_extent

    @property
    def y0_global(self):
        return int(self.y0 / self.yum + .5)

    @property
    def y1_global(self):
        return self.y0_global + self.directory.y_extent

    @property
    def z0_global(self):
        return int(self.z0 / self.zum + .5)

    @property
    def z1_global(self):
        return self.z0_global + self.directory.z_extent

    @staticmethod
    def find_volumes(volumes:typing.Sequence["StitchSrcVolume"],
                     x0:int, x1:int, y0:int, y1:int, z0:int, z1:int)\
            -> typing.Sequence["StitchSrcVolume"]:
        """
        Return the volumes that overlap a given block
        :param volumes: the candidate volumes
        :param x0: the block's x start in pixels
        :param x1: the block's x end
        :param y0: the block's y start
        :param y1: the block's y end
        :param z0: the block's z start
        :param z1: the block's z end
        :return: the volumes that overlap a given block
        """
        return [volume for volume in volumes
                if volume.does_overlap(x0, x1, y0, y1, z0, z1)]

    def get_mask(self, x0:int, x1:int, y0:int, y1:int, z0:int, z1:int)\
            -> np.ndarray:
        """
        Get mask of voxels that are within the volume. This assumes that
        self.does_overlap(x0, x1, y0, y1, z0, z1) is True

        :param x0: the x start of the block
        :param x1: the x end of the block
        :param y0: the y start of the block
        :param y1: the y end of the block
        :param z0: the z start of the block
        :param z1: the z end of the block
        :return: a boolean array of the size of the block where each array
        voxel is True if the voxel is within the volume.
        """
        x0r = self.x_relative(x0)
        y0r = self.y_relative(y0)
        z0r = self.z_relative(z0)
        x1r = x0r + x1 - x0
        y1r = y0r + y1 - y0
        z1r = z0r + z1 - z0
        z, y, x = np.mgrid[z0r:z1r, y0r:y1r, x0r:x1r]

        if self.is_oblique:
            mask = (x >= z) & (x - self.trailing_oblique_start < z)
        else:
            mask = (x >= 0) & (z < self.directory.x_extent)
        mask = mask & \
               (y >= 0) & (y < self.directory.y_extent) &\
               (z >= 0) & (z < self.directory.z_extent)
        return mask

    def distance_to_edge(self, x0: int, x1: int, y0: int, y1: int,
                         z0: int, z1: int)\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the distance to the nearest edge for every voxel in the block
        defined by x0:x1, y0:y1, z0:z1. Points outside of the volume are marked
        with negative distance.

        :param x0: the starting x, in global coordinates
        :param x1: the ending x, in global coordinates
        :param y0: the starting y, in global coordinates
        :param y1: the ending y, in global coordinates
        :param z0: the starting z, in global coordinates
        :param z1: the ending z in global coordinates
        :return: the distance to the nearest edge for every voxel in the block
        for the z, y and x directions
        """
        xs = x1 - x0
        ys = y1 - y0
        zs = z1 - z0
        z, y, x = np.mgrid[
                  self.z_relative(z0):self.z_relative(z0) + zs,
                  self.y_relative(y0):self.y_relative(y0) + ys,
                  self.x_relative(x0):self.x_relative(x0) + xs]
        if self.is_oblique:
            to_x0 = x - z
            to_x1 = self.directory.x_extent - x - z
        else:
            to_x0 = x
            to_x1 = self.directory.x_extent - x
        to_x = np.maximum(0, np.minimum(to_x0, to_x1))
        to_y0 = y
        to_y1 = self.directory.y_extent - y
        to_y = np.maximum(0, np.minimum(to_y0, to_y1))
        to_z0 = z
        to_z1 = self.directory.z_extent - z
        to_z = np.maximum(0, np.minimum(to_z0, to_z1))
        if not self.is_inside(x0, x1, y0, y1, z0, z1):
            mask = self.get_mask(x0, x1, y0, y1, z0, z1)
            to_x[~ mask] = -1
            to_y[~ mask] = -1
            to_z[~ mask] = -1
        return to_z, to_y, to_x

    def read_block(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int)\
            -> np.ndarray:
        x0r = self.x_relative(x0)
        y0r = self.y_relative(y0)
        z0r = self.z_relative(z0)
        x1r = x0r + x1 - x0
        y1r = y0r + y1 - y0
        z1r = z0r + z1 - z0
        block = np.zeros((z1 - z0, y1 - y0, x1 - x0), self.directory.dtype)
        x0a = max(0, x0r)
        x1a = min(self.directory.x_extent, x1r)
        y0a = max(0, y0r)
        y1a = min(self.directory.y_extent, y1r)
        z0a = max(0, z0r)
        z1a = min(self.directory.z_extent, z1r)
        x0s = (x0a // self.directory.x_block_size) * self.directory.x_block_size
        y0s = (y0a // self.directory.y_block_size) * self.directory.y_block_size
        z0s = (z0a // self.directory.z_block_size) * self.directory.z_block_size
        for xs, ys, zs in itertools.product(
            range(x0s, x1a, self.directory.x_block_size),
            range(y0s, y1a, self.directory.y_block_size),
            range(z0s, z1a, self.directory.z_block_size)):
            sub_block = self.directory.read_block(xs, ys, zs)
            if xs < x0a:
                sub_block = sub_block[:, :, x0a - xs:]
                x0f = x0a
            else:
                x0f = xs
            if ys < y0a:
                sub_block = sub_block[:, y0a - ys:]
                y0f = y0a
            else:
                y0f = ys
            if zs < z0a:
                sub_block = sub_block[z0a - zs:]
                z0f = z0a
            else:
                z0f = zs
            x1f = min(x0f + sub_block.shape[2], x1a)
            y1f = min(y0f + sub_block.shape[1], y1a)
            z1f = min(z0f + sub_block.shape[0], z1a)
            sub_block = sub_block[:z1f-z0f, :y1f-y0f, :x1f-x0f]
            block[z0f - z0r:z1f - z0r,
                  y0f - y0r:y1f - y0r,
                  x0f - x0r:x1f - x0r] = sub_block
        return block

    def align(self, other:"StitchSrcVolume",
              x:int, y:int, z:int,
              pad:typing.Tuple[int, int, int],
              sigma:typing.Tuple[float, float, float],
              border:typing.Tuple[int, int, int],
              max_iter=100) -> \
            typing.Tuple[float, typing.Tuple[float, float, float]]:
        """
        Align two volumes, centered around a position.

        :param other: the other volume to match against.
        :param x: x coordinate of initial position in pixels (global)
        :param y: y coordinate of initial position in pixels (global)
        :param z: z coordinate of initial position in pixels (global)
        :param pad: the half-size of the window in which we look
        :param sigma: the blurring sigma
        :param border: the amount we fetch, in addition to the padding to allow
        us to move without having to refetch
        :return: a tuple of pearson correlation coefficient and the adjusted
        position in the "other" volume
        """
        z0, y0, x0 = [a - b for a, b in zip((z, y, x), pad)]
        z1, y1, x1 = [a + b + 1for a, b in zip((z, y, x), pad)]
        if not self.is_inside(x0, x1, y0, y1, z0, z1) or \
            not other.is_inside(x0, x1, y0, y1, z0, z1):
            return 0, (z, y, x)
        fixed = ndimage.gaussian_filter(
            self.read_block(x0, x1, y0, y1, z0, z1).astype(np.float32),
            sigma=sigma)
        xm, ym, zm = x, y, z
        x0mb = y0mb = z0mb = x1mb = y1mb = z1mb = 0
        positions_seen = set((z, y, x))
        last_best = 0
        for iter in range(max_iter):
            z0m, y0m, x0m = [a - b - 1 for a, b in zip((zm, ym, xm), pad)]
            z1m, y1m, x1m = [a + b + 2 for a, b in zip((zm, ym, xm), pad)]
            if x0m < other.x0_global or x1m > other.x1_global or\
                y0m < other.y0_global or y1m > other.y1_global or\
                z0m < other.z0_global or z1m > other.z1_global:
                # we are at the border, no clear way to proceed.
                return last_best, [zm, ym, xm]
            if x0mb > x0m or y0mb > y0m or z0mb > z0m or \
                x1mb < x1m or y1mb < y1m or z1mb < z1m:
                # We need to read another window.
                window, (x0mb, x1mb, y0mb, y1mb, z0mb, z1mb) = \
                    other.read_window(x0m, x1m, y0m, y1m, z0m, z1m, border)
                moving = ndimage.gaussian_filter(window.astype(np.float32),
                                                 sigma=sigma)
            gradient = compute_pearson_gradient(
                fixed, moving[z0m-z0mb:z1m-z0mb,
                              y0m-y0mb:y1m-y0mb,
                              x0m-x0mb:x1m-x0mb])
            if np.all(np.isnan(gradient)):
                return last_best, (zm, ym, xm)
            last_best = np.nanmax(gradient)
            dz, dy, dx = np.argwhere(gradient == last_best)[0] - 1
            zm, ym, xm = zm + dz, ym + dy, xm + dx
            if (zm, ym, xm) in positions_seen:
                return last_best, (zm, ym, xm)
            positions_seen.add((zm, ym, xm))
        return last_best, (zm, ym, xm)

    def read_window(self, x0:int, x1:int, y0:int, y1:int, z0:int, z1:int,
                    border=typing.Tuple[int, int, int]) ->\
            typing.Tuple[np.ndarray,
                         typing.Tuple[int, int, int, int, int, int]]:
        """
        Try to read a window with a border, at least reading the minimum
        volume entered.

        :param x0: global x min position that must be read.
        :param x1: global x max position that must be read.
        :param y0: global y min position that must be read.
        :param y1: global y max position that must be read.
        :param z0: global z min position that must be read.
        :param z1: global z max position that must be read.
        :param border: the optimal border to be read, in z, y, x form
        :return: a two tuple:
            the array returned,
            a six-tuple of x0, x1, y0, y1, z0, z1 which are the global
            coordinates of the array
        """
        x0a = max(x0 - border[2], self.x0_global)
        x1a = min(x1 + border[2], self.x1_global)
        y0a = max(y0 - border[1], self.y0_global)
        y1a = min(y1 + border[1], self.y1_global)
        z0a = max(z0 - border[1], self.z0_global)
        z1a = min(z1 + border[1], self.z1_global)
        return self.read_block(x0a, x1a, y0a, y1a, z0a, z1a),\
               (x0a, x1a, y0a, y1a, z0a, z1a)


def compute_pearson_gradient(fixed:np.ndarray, moving:np.ndarray)->np.ndarray:
    """
    Compute the Pearson Correlation Coefficient at a central point and all
    26 points immediately surrounding it

    :param fixed: the fixed array
    :param moving: the moving array which must be +2 bigger in each direction,
    centered at +1 offset
    :return: a 3x3x3 array of correlation coefficients at each direction
    """
    gradient = np.zeros((3, 3, 3))
    for i, j, k in itertools.product((0, 1, 2), (0, 1, 2), (0, 1, 2)):
        gradient[i, j, k] = np.corrcoef(
            fixed.flatten(),
            moving[i:i+fixed.shape[0],
                   j:j+fixed.shape[1],
                   k:k+fixed.shape[2]].flatten())[0, 1]
    return gradient

# The global list of volumes goes here - a dictionary of UUID to volume
VOLUMES:typing.Dict[uuid.UUID, StitchSrcVolume] = None

# The output volume directory goes here
OUTPUT:Directory = None

def set_volumes_and_output(volumes, output):
    global VOLUMES, OUTPUT
    VOLUMES=volumes
    OUTPUT=output

def do_block(x0: int, y0:int, z0: int, x0g:int, y0g:int, z0g:int):
    zs, ys, xs = OUTPUT.get_block_size(x0-x0g, y0-y0g, z0-z0g)
    x1 = x0 + xs
    y1 = y0 + ys
    z1 = z0 + zs
    volumes = [volume for volume in VOLUMES
               if volume.does_overlap(x0, x1, y0, y1, z0, z1)]
    if len(volumes) == 0:
        return
    elif len(volumes) == 1:
        volume = volumes[0]
        block = volume.read_block(x0, x1, y0, y1, z0, z1)
    else:
        block = np.zeros((z1-z0, y1-y0, x1-x0), OUTPUT.dtype)
        distances = [volume.distance_to_edge(x0, x1, y0, y1, z0, z1)
                     for volume in volumes]
        masks = [volume.get_mask(x0, x1, y0, y1, z0, z1)
                 for volume in volumes]
        for volume, (dz, dy, dx), mask in zip(volumes, distances, masks):
            vblock = volume.read_block(x0, x1, y0, y1, z0, z1)
            fraction = volume.get_mask(x0, x1, y0, y1, z0, z1).astype(float)
            for idx in [_ for _ in range(len(volumes))
                        if volumes[_].key != volume.key]:
                other = volumes[idx]
                other_mask = masks[idx]
                both_mask = mask & other_mask
                other_dz, other_dy, other_dx = distances[idx]
                distance = 100000000 * np.ones(dx.shape, np.float32)
                other_distance = 100000000 * np.ones(other_dx.shape, np.float32)
                if np.any(dx[both_mask] != other_dx[both_mask]):
                    distance = dx
                    other_distance = other_dx
                if np.any(dy[both_mask] != other_dy[both_mask]):
                    distance = np.minimum(distance, dy)
                    other_distance = np.minimum(other_distance, other_dy)
                if np.any(dz[both_mask] != other_dz[both_mask]):
                    distance = np.minimum(distance, dz)
                    other_distance = np.minimum(other_distance, other_dz)
                angle = np.arctan2(distance[both_mask],
                                   other_distance[both_mask])
                blending = np.sin(angle) ** 2
                fraction[both_mask] *= blending
            block += (fraction * vblock).astype(block.dtype)

    OUTPUT.write_block(block, x0-x0g, y0-y0g, z0-z0g)


def get_output_size(volumes:typing.Sequence[StitchSrcVolume])\
        -> typing.Tuple[int, int, int]:
    """
    Calculate the size of the output volume

    :param volumes: the input volumes
    :return: the output volume in voxels, a tuple of z, y and x
    """
    x0 = [int(volume.x0 / volume.xum + .5)  for volume in volumes]
    x1 = [x0i + volume.directory.x_extent for x0i, volume in
          zip(x0, volumes)]
    y0 = [int(volume.y0 / volume.yum + .5)  for volume in volumes]
    y1 = [y0i + volume.directory.y_extent for y0i, volume in
          zip(y0, volumes)]
    z0 = [int(volume.z0 / volume.zum + .5)  for volume in volumes]
    z1 = [z0i + volume.directory.z_extent for z0i, volume in
          zip(z0, volumes)]
    if len(x0) == 1:
        return z1[0] - z0[0], y1[0] - y0[0], x1[0] - x0[0]
    return max(*z1) - min(*z0), max(*y1) - min(*y0), max(*x1) - min(*x0)


def run(volumes:typing.Sequence[StitchSrcVolume], output: Directory,
        x0g:int, y0g:int, z0g:int,
        n_workers:int, silent=False):
    global VOLUMES, OUTPUT, OFFSET
    VOLUMES = volumes
    OUTPUT = output
    xr = range(x0g, x0g + output.x_extent, output.x_block_size)
    yr = range(y0g, y0g + output.y_extent, output.y_block_size)
    zr = range(z0g, z0g + output.z_extent, output.z_block_size)
    futures = []
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            for x0, y0, z0 in itertools.product(xr, yr, zr):
                futures.append(
                    pool.apply_async(do_block, (x0, y0, z0, x0g, y0g, z0g)))

            for future in tqdm.tqdm(futures,
                                    desc="Writing blocks"):
                future.get()
            OUTPUT.close()
    else:
        for x0, y0, z0 in tqdm.tqdm(
                itertools.product(xr, yr, zr),
                total=len(xr) * len(yr) * len(zr)):
            do_block(x0, y0, z0, x0g, y0g, z0g)
        OUTPUT.close()