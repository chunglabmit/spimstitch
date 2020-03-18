# -*- coding: utf-8 -*-

from blockfs.directory import Directory
import itertools
import multiprocessing
import numpy as np
import os
import tqdm
import typing
import uuid


class StitchSrcVolume:

    def __init__(self, path:str, x_step_size:float, yum:float):
        """

        :param path: Path to a blockfs directory file. The path contains
        position metadata for the volume - the path should be something like
        %d_%d/1_1_1/precomputed.blockfs
        :param x_step_size: stepper size in the X direction. We assume that the
        X pixel size is sqrt(2) * x_step_size, so the Z pixel size is the
        same as the stepper size.
        :param yum: Y pixel size of pixels in the original planes
        """
        self.key = uuid.uuid4()
        self.path = path
        self.directory = Directory.open(path)
        self.xum = x_step_size
        self.zum = x_step_size
        self.yum = yum
        metadata_dir = os.path.split(os.path.dirname(os.path.dirname(path)))[-1]
        self.x0, self.y0 = [int(_) / 10 for _ in metadata_dir.split("_")]
        self.z0 = 0

    def rebase(self, x0:float, y0:float)->type(None):
        """
        Once we find out where the top left corner is, we rebase all the
        stacks so that the first one starts at 0.

        :param x0: The absolute left of all of the stacks
        :param y0: The absolute top of all of the stacks
        """
        self.x0 -= x0
        self.y0 -= y0

    @staticmethod
    def rebase_all(volumes:typing.Sequence["StitchSrcVolume"]):
        """
        Rebase all the volumes so that the whole space starts at zero

        :param volumes: the volumes to be rebased
        """
        x0 = volumes[0].x0
        y0 = volumes[0].y0
        for volume in volumes[1:]:
            x0 = min(x0, volume.x0)
            y0 = min(y0, volume.y0)
        for volume in volumes:
            volume.rebase(x0, y0)

    @property
    def trailing_oblique_start(self):
        return self.directory.x_extent - self.directory.z_extent

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
        #
        # The corner cases
        #
        if x0_relative < z0_relative:
            return False # top corner of block is below the leading oblique
       # bottom corner of block is above the trailing oblique
        return x1_relative - self.trailing_oblique_start < z1_relative

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
        mask = np.ones((z1 - z0, y1 - y0, x1 - x0), bool)
        x0r = self.x_relative(x0)
        y0r = self.y_relative(y0)
        z0r = self.z_relative(z0)
        x1r = x0r + x1 - x0
        y1r = y0r + y1 - y0
        z1r = z0r + z1 - z0
        z, y, x = np.mgrid[z0r:z1r, y0r:y1r, x0r:x1r]

        mask = (x >= z) & (x - self.trailing_oblique_start < z) &\
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
        to_x0 = x - z
        to_x1 = self.directory.x_extent - x - z
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
            block[z0f - z0a:z1f - z0a,
                  y0f - y0a:y1f - y0a,
                  x0f - x0a:x1f - x0a] = sub_block
        return block

# The global list of volumes goes here - a dictionary of UUID to volume
VOLUMES:typing.Dict[uuid.UUID, StitchSrcVolume] = None

# The output volume directory goes here
OUTPUT:Directory = None


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