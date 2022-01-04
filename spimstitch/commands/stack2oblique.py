import argparse
import glob
import logging
import multiprocessing
import os

import numpy as np
from blockfs.directory import Directory
from precomputed_tif.blockfs_stack import BlockfsStack
import sys

from precomputed_tif.ngff_stack import NGFFStack

from spimstitch.ngff import NGFFDirectory
from ..oblique import spim_to_blockfs, get_blockfs_dims
from ..stack import SpimStack, StackFrame


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Convert a stack of images taken at a 45° bias to a "
                    "precomputed blockfs volume"
    )
    parser.add_argument(
        "--input",
        help="A glob expression to collect all of the stack .tif files",
        required=True)
    parser.add_argument(
        "--output",
        help="The root directory for the precomputed volume",
        required=True)
    parser.add_argument(
        "--levels",
        help="The number of mipmap levels in the precomputed volume",
        default=5,
        type=int)
    parser.add_argument(
        "--x-step-size",
        help="The size of a stage motion step in microns",
        type=float,
        default=3.625 / np.sqrt(2)
    )
    parser.add_argument(
        "--y-voxel-size",
        help="The size of a voxel in the plane in microns",
        type=float,
        default=3.625
    )
    parser.add_argument(
        "--log-level",
        help="The log level for logging",
        default="WARNING")
    parser.add_argument(
        "--n-writers",
        help="The number of writer processes for writing blockfs files",
        default=min(12, os.cpu_count()),
        type=int)
    parser.add_argument(
        "--n-workers",
        help="The number of worker processes for the processing pipeline",
        default=min(12, os.cpu_count()),
        type=int)
    parser.add_argument(
        "--ngff",
        help="If present, output NGFF instead of blockfs",
        action="store_true"
    )
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    logging.basicConfig(level=getattr(logging, opts.log_level))
    paths = sorted(glob.glob(opts.input))
    if len(paths) == 0:
        print("No files were found for expression: %s" % opts.input)
        return
    frame = StackFrame(paths[0], 0, 0, 0)
    stack = SpimStack(paths, frame.x0, frame.x1, frame.y0, frame.y1, 0)
    zs, ys, xs, dtype = get_blockfs_dims(stack)
    if opts.ngff:
        bfs_stack = NGFFStack((zs, ys, xs), opts.output)
        bfs_stack.create()
    else:
        bfs_stack = BlockfsStack((zs, ys, xs), opts.output)
    x_step_size = opts.x_step_size
    y_voxel_size = opts.y_voxel_size
    z_voxel_size = opts.y_voxel_size / np.sqrt(2)
    bfs_stack.write_info_file(
        opts.levels, voxel_size=(x_step_size, y_voxel_size, z_voxel_size))
    if opts.ngff:
        directory = NGFFDirectory(bfs_stack)
        directory.create()
    else:
        bfs_level1_dir = os.path.join(
            opts.output, "1_1_1", BlockfsStack.DIRECTORY_FILENAME)
        if not os.path.exists(os.path.dirname(bfs_level1_dir)):
            os.mkdir(os.path.dirname(bfs_level1_dir))
        directory = Directory(xs,
                              ys,
                              zs,
                              dtype,
                              bfs_level1_dir,
                              n_filenames=opts.n_writers)
        directory.create()
        directory.start_writer_processes()
    spim_to_blockfs(stack, directory, opts.n_workers,
                    voxel_size = y_voxel_size, x_step_size=x_step_size)
    for level in range(2, opts.levels+1):
        bfs_stack.write_level_n(level, n_cores=opts.n_writers)


if __name__=="__main__":
    main()