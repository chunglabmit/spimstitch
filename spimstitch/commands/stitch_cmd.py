import argparse
import json

import typing
from blockfs.directory import  Directory
import logging
from precomputed_tif.blockfs_stack import BlockfsStack
import os
import sys

from ..stitch import get_output_size, StitchSrcVolume, run


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="The root directory of the oblique volume tree. The program expects "
        "blockfs Neuroglancer volumes in directories whose name is in the "
        "format, <x>_<y> where <x> and <y> are the X and Y coordinates of "
        "the top left corner of the volume.",
        required=True)
    parser.add_argument(
        "--output",
        help="The directory for the precomputed volume output"
    )
    parser.add_argument(
        "--levels",
        help="The number of mipmap levels in the precomputed volume",
        default=5,
        type=int)
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
        "--silent",
        help="Turn off progress bars",
        action="store_true")
    parser.add_argument(
        "--x-step-size",
        help="X step size in microns",
        default=1.28,
        type=float)
    parser.add_argument(
        "--y-voxel-size",
        help="Size of a voxel in the Y direction in microns",
        default=1.8,
        type=float)
    parser.add_argument(
        "--z-offset",
        help="# of voxels of offset between the start of the stack above "
             "in Z and the stack underneath it",
        default=2048,
        type=int
    )
    parser.add_argument(
        "--output-size",
        help="Size of the output volume (x,y,z). Defaults to the extent of all "
             "prestitched volumes.")
    parser.add_argument(
        "--output-offset",
        help="Offset of the output volume. Only use with --output-size. ")
    parser.add_argument(
        "--alignment",
        help="Alignment file from oblique-align. Default is use static "
        "alignment"
    )
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    logging.basicConfig(level=getattr(logging,opts.log_level))
    volume_paths = []
    zs = []

    for root, folders, files in os.walk(opts.input, followlinks=True):
        if os.path.split(root)[-1] == "1_1_1":
            for file in files:
                if file == BlockfsStack.DIRECTORY_FILENAME:
                    volume_paths.append(os.path.join(root, file))
                    zs.append(os.path.split(os.path.dirname(root))[1])
    all_z = sorted(set(zs))
    z_offsets = [opts.z_offset * all_z.index(z) for z in zs]
    volumes = [
        StitchSrcVolume(volume_path,
                        opts.x_step_size,
                        opts.y_voxel_size,
                        z_offset)
        for volume_path, z_offset in zip(volume_paths, z_offsets)]
    adjust_alignments(opts, volumes)
    StitchSrcVolume.rebase_all(volumes)
    if opts.output_size is None:
        zs, ys, xs = get_output_size(volumes)
        x0 = y0 = z0 = 0
    else:
        xs, ys, zs = [int(_) for _ in opts.output_size.split(",")]
        if opts.output_offset is None:
            x0 = y0 = z0 = 0
        else:
            x0, y0, z0 = [int(_) for _ in opts.output_offset.split(",")]
    if not os.path.exists(opts.output):
        os.mkdir(opts.output)
    l1_dir = os.path.join(opts.output, "1_1_1")
    if not os.path.exists(l1_dir):
        os.mkdir(l1_dir)
    output = BlockfsStack((zs, ys, xs), opts.output)
    voxel_size = (opts.x_step_size * 1000,
                  opts.y_voxel_size * 1000,
                  opts.x_step_size * 1000)
    output.write_info_file(opts.levels, voxel_size)
    directory_path = os.path.join(l1_dir, BlockfsStack.DIRECTORY_FILENAME)
    directory = Directory(xs, ys, zs, volumes[0].directory.dtype,
                          directory_path,
                          n_filenames=opts.n_writers)
    directory.create()
    directory.start_writer_processes()
    run(volumes, directory, x0, y0, z0, opts.n_workers, opts.silent)
    directory.close()
    for level in range(2, opts.levels + 1):
        output.write_level_n(level, opts.silent, opts.n_writers)


def adjust_alignments(opts, volumes:typing.Sequence[StitchSrcVolume]):
    """
    Adjust the volume coordinates based on alignments recorded by
    oblique-align or similar.

    :param opts: The command-line options - we take the --alignment arg
    as a json file.
    :param volumes: The volumes to be adjusted
    """
    if opts.alignment is not None:
        alignments = {}
        with open(opts.alignment) as fd:
            d = json.load(fd)
        if "alignments" in d:
            for k, v in d["alignments"].items():
                alignments[tuple(json.loads(k)[:-1])] = v
        for volume in volumes:
            k = (volume.x0, volume.y0)
            if k in alignments:
                volume.x0, volume.y0, _ = alignments[k]


if __name__ == "__main__":
    main()
