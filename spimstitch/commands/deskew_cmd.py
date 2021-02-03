import argparse
import itertools
import multiprocessing
import pathlib

import numpy as np
import sys
import typing
import tqdm
from scipy import ndimage
from precomputed_tif.client import ArrayReader
from precomputed_tif.blockfs_stack import BlockfsStack
from blockfs.directory import Directory


def parse_args(args:typing.Sequence[str]=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description=
        "The oblique microscope acquires the volume as a "
        "parellelopiped with the X/Z direction squished "
        "by a factor of sqrt(2) * x-step-size / y-step-size. "
        "This program unsquishes for the cases where x-step-size and "
        "y-step-size are not well-matched."
    )
    parser.add_argument(
        "--input",
        help="Location of input blockfs neuroglancer volume",
        required=True)
    parser.add_argument(
        "--output",
        help="Location for output volume",
        required=True
    )
    parser.add_argument(
        "--x-step-size",
        help="The X step size, e.g. from the metadata.txt file.",
        required=True,
        type=float
    )
    parser.add_argument(
        "--y-voxel-size",
        help="The Y voxel size, e.g. from the metadata.txt file.",
        required=True,
        type=float
    )
    parser.add_argument(
        "--levels",
        help="# of pyramid levels in the output file",
        default=5,
        type=int
    )
    parser.add_argument(
        "--n-cores",
        help="The number of CPUs used when reading and warping",
        default=multiprocessing.cpu_count(),
        type=int
    )
    parser.add_argument(
        "--n-writers",
        help="The number of writer processes",
        default=min(11, multiprocessing.cpu_count()),
        type=int
    )
    return parser.parse_args(args)


ARRAY_READER:ArrayReader=None
DIRECTORY:Directory=None


def xz2xd(x:typing.Union[int, float, np.ndarray],
          z:typing.Union[int, float, np.ndarray],
          r:float):
    """
    Convert from source to destination coordinate system

    :param x: x coordinate or coordinates in the parallelopiped volume space
    :param z: z coordinate or coordinates in the parallelopiped volume space
    :param r: Ratio of step size to voxel size
    :return: x coordinate or coordinates in the cubic output volume space
    """
    r2 = np.sqrt(r)
    return (1/r + 1) * x + (1/r - 1) * z


def xz2zd(x:typing.Union[int, float, np.ndarray],
          z:typing.Union[int, float, np.ndarray],
          r:float):
    """
    Convert from source to destination coordinate system

    :param x: x coordinate or coordinates in the parallelopiped volume space
    :param z: z coordinate or coordinates in the parallelopiped volume space
    :param r: Ratio of step size to voxel size
    :return: z coordinate or coordinates in the cubic output volume space
    """
    return (1/r + 1) * z + (1/r - 1) * x

# This is the math for back-converting the coordinates
#
# a b   (1/r2 + r2)  (1/r2 - r2)
# c d = (1/r2 - r2)  (1/r2 + r2)
#
# det = (ad - bc) = (1/r2 + r2)(1/r2 + r2) - (1/r2 - r2)(1/r2 - r2) = (1/r + 2 + r) - (1/r - 2 + r) = 4
#
# inv = d  -b  / det
#       -c  a
#     = (r2 + 1/r2) / 4    (r2 - 1/r2) / 4
#       (r2 - 1/r2) / 4    (r + 1/r2) / 4


def xdzd2x(xd:typing.Union[int, float, np.ndarray],
           zd:typing.Union[int, float, np.ndarray],
           r:float):
    """
    Convert from cuboid (dest) coordinate system to parallelopiped (source)
    :param xd: x coordinate in the destination coordinate system
    :param zd: z coordinate in the destination coordinate system
    :param r: the ratio between step size and voxel size
    :return: converted x coordinate or coordinates
    """
    return xd * (r + 1) / 4 + zd * (r - 1) / 4


def xdzd2z(xd:typing.Union[int, float, np.ndarray],
           zd:typing.Union[int, float, np.ndarray],
           r:float):
    """
    Convert from cuboid (dest) coordinate system to parallelopiped (source)
    :param xd: x coordinate in the destination coordinate system
    :param zd: z coordinate in the destination coordinate system
    :param r: the ratio between step size and voxel size
    :return: converted z coordinate or coordinates
    """
    r2 = np.sqrt(r)
    return xd * (r - 1) / 4 + zd * (r + 1) / 4


def do_one(x0:int, y0:int, z0:int, xoff:int, zoff:int, r:float) ->\
        typing.NoReturn:
    """
    Process one block
    :param x0: x start of block
    :param y0: y start of block
    :param z0: z start of block
    :param xoff: offset from destination coords to block coords in X
    :param zoff: offset from destination coords to block coords in Y
    :param r: the distance ratio in the X+Z direction wrt the X-Z direction
    """
    zsize, ysize, xsize = DIRECTORY.get_block_size(x0, y0, z0)
    x1, y1, z1 = x0+xsize, y0+ysize, z0+zsize
    xd = np.arange(x0, x1)
    zd = np.arange(z0, z1)
    o = np.zeros((len(zd), len(xd)), int)
    xd = o + xd.reshape(1, -1)
    zd = o + zd.reshape(-1, 1)
    xs = xdzd2x(xd+xoff, zd+zoff, r)
    zs = xdzd2z(xd+xoff, zd+zoff, r)
    xs0, xs1 = int(np.min(xs)) - 1, int(np.ceil(np.max(xs))) + 1
    zs0, zs1 = int(np.min(zs)) - 1, int(np.ceil(np.max(zs))) + 1
    if xs1 <= 0 or zs1 <= 0 or \
            xs0 >= ARRAY_READER.shape[2] or zs0 >= ARRAY_READER.shape[0]:
        return
    if xs0 < 0 or zs0 < 0:
        # Negative indexes not allowed.
        xs0a = max(0, xs0)
        zs0a = max(0, zs0)
        src_block = np.zeros((zs1-zs0, y1-y0, xs1-xs0), ARRAY_READER.dtype)
        src_block[zs0a-zs0:zs1-zs0, :, xs0a-xs0:xs1-xs0] = \
            ARRAY_READER[zs0a:zs1, y0:y1, xs0a:xs1]
    else:
        src_block = ARRAY_READER[zs0:zs1, y0:y1, xs0:xs1]
    if np.all(src_block == 0):
        return
    dest_block = np.zeros((zsize, ysize, xsize), ARRAY_READER.dtype)
    for y in range(y0, y1):
        dest_block[:, y - y0, :] = ndimage.map_coordinates(
            src_block,
            (zs - zs0, y - y0 + o, xs - xs0))
    DIRECTORY.write_block(dest_block, x0, y0, z0)


def main(args:typing.Sequence[str]=sys.argv[1:]):
    global ARRAY_READER, DIRECTORY
    opts = parse_args(args)
    x_step_size = opts.x_step_size
    y_voxel_size = opts.y_voxel_size
    r = x_step_size / y_voxel_size * np.sqrt(2)
    ARRAY_READER = ArrayReader(
        pathlib.Path(opts.input).as_uri(), format="blockfs")
    dest_path = pathlib.Path(opts.output)
    dest_path.mkdir(parents=True, exist_ok=True)
    #
    # Find the size of the destination volume by looking
    # at the x/z corners
    #
    x00d = xz2xd(0, 0, r)
    x01d = xz2xd(0, ARRAY_READER.shape[0], r)
    x10d = xz2xd(ARRAY_READER.shape[2], 0, r)
    x11d = xz2xd(ARRAY_READER.shape[2], ARRAY_READER.shape[0], r)
    z00d = xz2zd(0, 0, r)
    z01d = xz2zd(0, ARRAY_READER.shape[0], r)
    z10d = xz2zd(ARRAY_READER.shape[2], 0, r)
    z11d = xz2zd(ARRAY_READER.shape[2], ARRAY_READER.shape[0], r)
    x0d = int(np.min([x00d, x01d, x10d, x11d]))
    x1d = int(np.ceil(np.max([x00d, x01d, x10d, x11d])))
    z0d = int(np.min([z00d, z01d, z10d, z11d]))
    z1d = int(np.ceil(np.max([z00d, z01d, z10d, z11d])))
    output_shape = (z1d - z0d, ARRAY_READER.shape[1], x1d - x0d)
    #
    # Get the blockfs destination started
    #
    blockfs_stack = BlockfsStack(output_shape, opts.output)
    voxel_size = (1000. * x_step_size * np.sqrt(2),
                  1000. * y_voxel_size,
                  1000. * x_step_size * np.sqrt(2))
    blockfs_stack.write_info_file(opts.levels, voxel_size)
    bfs_level1_dir = \
        pathlib.Path(opts.output) / "1_1_1" / BlockfsStack.DIRECTORY_FILENAME
    bfs_level1_dir.parent.mkdir(parents=True, exist_ok=True)
    DIRECTORY = Directory(output_shape[2],
                          output_shape[1],
                          output_shape[0],
                          ARRAY_READER.dtype,
                          str(bfs_level1_dir),
                          n_filenames=opts.n_writers)
    DIRECTORY.create()
    DIRECTORY.start_writer_processes()
    xds = np.arange(0, output_shape[2], DIRECTORY.x_block_size)
    yds = np.arange(0, output_shape[1], DIRECTORY.y_block_size)
    zds = np.arange(0, output_shape[0], DIRECTORY.z_block_size)
    with multiprocessing.Pool(opts.n_cores) as pool:
        futures = []
        for x0di, y0di, z0di in itertools.product(xds, yds, zds):
            futures.append(pool.apply_async(
                do_one, (
                    x0di, y0di, z0di, x0d, z0d, r
                )
            ))
        for future in tqdm.tqdm(futures):
            future.get()
        DIRECTORY.close()
    for level in range(2, opts.levels+1):
        blockfs_stack.write_level_n(level, n_cores=opts.n_writers)


if __name__ == "__main__":
    main()
