import os

import multiprocessing

import argparse
import pathlib
import numpy as np
import sys
import typing
from blockfs import Directory
from precomputed_tif.blockfs_stack import BlockfsStack

from ..stitch import StitchSrcVolume, run, get_output_size, do_block


def parse_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Root input directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for neuroglancer volume"
    )
    parser.add_argument(
        "--voxel-size",
        default=1.8,
        type=float,
        help="Size of camera voxel in microns"
    )
    parser.add_argument(
        "--x-step-size",
        default=1.28,
        type=float,
        help="Size of one x-stepper step in microns"
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=5,
        help="# of neuroglancer mipmap levels"
    )
    parser.add_argument(
        "--n-cores",
        default=min(20, multiprocessing.cpu_count()),
        type=int,
        help="# of processors to use in multiprocessing"
    )
    parser.add_argument(
        "--n-writers",
        default=min(11, multiprocessing.cpu_count()),
        type=int,
        help="# of processors to devote to writing blockfs"
    )
    parser.add_argument(
        "--compression",
        default=3,
        type=int,
        help="Compression for tiff files, default=3, 0=none"
    )
    parser.add_argument("--silent",
                        action="store_true",
                        help="Don't print progress bar")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_arguments(args)
    directories = list(pathlib.Path(opts.input)
        .glob("*/*/*/1_1_1/precomputed.blockfs"))
    if len(directories) == 0:
        print("Could not find any subvolumes in %s" % opts.input)
        sys.exit(-1)
    volumes:typing.Dict[typing.Tuple[int, int, int], StitchSrcVolume] = {}
    xs = set()
    ys = set()
    zs = set()
    for directory in directories:
        z_path = directory.parent.parent
        z = int(z_path.name) / 10
        x_y_path = z_path.parent
        x, y = [int(_) / 10 for _ in x_y_path.name.split("_")]
        xs.add(x)
        ys.add(y)
        zs.add(z)
        volumes[x, y, z] = StitchSrcVolume(
            os.fspath(directory),
            opts.x_step_size,
            opts.voxel_size,
            0,
            is_oblique=False)
    #
    # The precomputed stacks have their X in the Z direction
    # and their Z in the X direction. xum, yum and zum are
    # in that orietation.
    xum = opts.voxel_size / np.sqrt(2)
    yum = opts.voxel_size
    zum = opts.x_step_size
    for (x, y, z), volume in volumes.items():
        volume.x0 = z
        volume.xum = xum
        volume.y0 = y
        volume.yum = yum
        #
        # Here, we offset old-x / new-z by the z-step to account
        # for the oblique slant which requires us to match lower
        # z top to upper z bottom.
        volume.z0 = x - z
        volume.zum = zum

    all_volumes = list(volumes.values())
    any_volume = all_volumes[0]
    StitchSrcVolume.rebase_all(all_volumes, z_too=True)
    z_extent, y_extent, x_extent = get_output_size(all_volumes)
    do_debug = False
    if do_debug:
        opts.output = "/tmp/spimstitch_debug"
    if not os.path.exists(opts.output):
        os.mkdir(opts.output)
    l1_dir = os.path.join(opts.output, "1_1_1")
    if not os.path.exists(l1_dir):
        os.mkdir(l1_dir)
    output = BlockfsStack((z_extent, y_extent, x_extent), opts.output)
    voxel_size = (opts.voxel_size * 1000,
                  opts.voxel_size * 1000,
                  opts.x_step_size * 1000)
    output.write_info_file(opts.levels, voxel_size)
    directory_path = os.path.join(l1_dir, BlockfsStack.DIRECTORY_FILENAME)
    directory = Directory(x_extent, y_extent, z_extent,
                          any_volume.directory.dtype,
                          directory_path,
                          n_filenames=opts.n_writers)
    directory.create()
    directory.start_writer_processes()
    if do_debug:
        from ..stitch import set_volumes_and_output
        set_volumes_and_output(all_volumes, directory)
        x, y, z = 1616, 772, 288
        x, y, z = [(_//64) * 64 for _ in (x, y, z)]
        do_block(x, y, z, 0, 0, 0)
        return
    run(all_volumes, directory, 0, 0, 0, opts.n_cores, opts.silent)
    directory.close()
    for level in range(2, opts.levels + 1):
        output.write_level_n(level, opts.silent, opts.n_writers)


if __name__ == "__main__":
    main()
