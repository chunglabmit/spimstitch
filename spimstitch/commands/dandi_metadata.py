"""
DANDI-related metadata

This command has a variety of subcommands
dandi-metadata target-file \
    --subject <subject> \
    [--run <run>] \
    --source <source-path>\
    --sample <slab-number> \
    --stain <stain-name> \
    --chunk <chunk-name>

This generates a path to the file to store the volume and its sidecar metadata.
<subject> is the subject's name
<run> is the run number. If not present, "1" is used.
<slab-number> is the slab number which is the sample number
<source-path> is the path to the directory holding the raw images
              (2021_blah-blah, not Ex_blah blah).
<stain> is the name of the stain.
<chunk-name> is the name of the stack chunk being saved.

The command prints out the path name relative to the subject directory
"""
import argparse
import json
import re
import pathlib
import shutil
import sys

from math import sqrt
from precomputed_tif.client import ArrayReader


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_target_file_parser(subparsers)
    build_write_sidecar_parser(subparsers)
    build_x_step_size_parser(subparsers)
    build_y_voxel_size_parser(subparsers)
    build_transform_parser(subparsers)
    build_order_dcimg_files(subparsers)
    build_rewrite_transforms(subparsers)
    return parser.parse_args(args)


def build_y_voxel_size_parser(subparsers):
    subparser = subparsers.add_parser("get-y-voxel-size")
    subparser.set_defaults(func=get_y_voxel_size)
    subparser.add_argument(
        "metadata_file",
        help="Path to the metadata.txt file"
    )


def build_x_step_size_parser(subparsers):
    subparser = subparsers.add_parser("get-x-step-size")
    subparser.set_defaults(func=get_x_step_size)
    subparser.add_argument(
        "metadata_file",
        help="Path to the metadata.txt file"
    )


def build_write_sidecar_parser(subparsers):
    subparser = subparsers.add_parser("write-sidecar")
    subparser.set_defaults(func=write_sidecar)
    subparser.add_argument(
        "--template",
        help="Path to template JSON file which serves as the base for"
             "the sidecar",
        required=True
    )
    subparser.add_argument(
        "--metadata-file",
        help="Path to the metadata file containing the oblique spim recorded "
             "metadata. Defaults to \"metadata.txt\" in the same "
             "directory as the volume"
    )
    subparser.add_argument(
        "--volume",
        help="Pointer to the NGFF volume",
        required=True
    )
    subparser.add_argument(
        "--volume-format",
        help="Precomputed format of the volume, e.g. \"blockfs\" or \"ngff\".",
        default="ngff"
    )
    subparser.add_argument(
        "--stain",
        help="Stain used for volume",
        required=True
    )
    subparser.add_argument(
        "--output",
        help="Name of the sidecar JSON file",
        required=True
    )


def build_target_file_parser(subparsers):
    subparser = subparsers.add_parser("target-file")
    subparser.set_defaults(func=target_file)
    subparser.add_argument(
        "--subject",
        help="Subject of the experiment e.g. the person or animal that "
             "the tissue came from",
        required=True
    )
    subparser.add_argument(
        "--sample",
        help="The ID for the tissue or slab being imaged",
        required=True
    )
    subparser.add_argument(
        "--run",
        help="The run number for the slab's imaging session",
        default="1"
    )
    subparser.add_argument(
        "--source-path",
        help="The path to the raw image's directory. This is used to extract "
             "the session name",
        required=True
    )
    subparser.add_argument(
        "--stain",
        help="The stain of the imaged channel",
        required=True
    )
    subparser.add_argument(
        "--chunk",
        help="The name of the stack being imaged",
        required=True
    )


def build_transform_parser(subparsers):
    subparser = subparsers.add_parser("write-transform")
    subparser.set_defaults(func=write_transform)
    subparser.add_argument(
        "--input",
        help="The path to the DCIMG file being converted",
        required=True)
    subparser.add_argument(
        "--output",
        help="The file to be written",
        required=True
    )
    subparser.add_argument(
        "--target-reference-frame",
        help="The name of the target reference frame, e.g. the slab #",
        required=True
    )
    subparser.add_argument(
        "--y-voxel-size",
        help="The size of a voxel in microns",
        default=1.8,
        type=float
    )
    subparser.add_argument(
        "dcimg_files",
        nargs="*",
        default=[],
        help="The remainder of the files should include all of the "
             "DCIMG files in the volume"
    )


def build_rewrite_transforms(subparsers):
    subparser = subparsers.add_parser("rewrite-transforms")
    subparser.set_defaults(func=rewrite_transforms)
    subparser.add_argument(
        "--align-file",
        help="The output from oblique-align",
        required=True
    )
    subparser.add_argument(
        "transform_files",
        nargs="*",
        default=[],
        help="The transform files output by the write-transforms subcommand"
    )

def build_order_dcimg_files(subparsers):
    subparser = subparsers.add_parser("order-dcimg-files")
    subparser.set_defaults(func=order_dcimg_files_cmd)
    subparser.add_argument(
        "dcimg_files",
        nargs="*",
        default=[],
        help="The remainder of the files should include all of the "
             "DCIMG files in the volume"
    )


def get_x_step_size(opts):
    metadata_path = pathlib.Path(opts.metadata_file)
    x_step_size, y_voxel_size = get_sizes(metadata_path)
    print(x_step_size)


def get_y_voxel_size(opts):
    metadata_path = pathlib.Path(opts.metadata_file)
    x_step_size, y_voxel_size = get_sizes(metadata_path)
    print(y_voxel_size)


def target_file(opts):
    source_path = pathlib.Path(opts.source_path)
    match = re.search("^(\\d{8})_(\\d{2})_(\\d{2})_(\\d{2})", source_path.name)
    if not match:
        print("%s did not match a date" % source_path.name,
              file=sys.stderr)
        exit(-1)
    session_groups = tuple(match.groups())
    dir_path = pathlib.Path("sub-%s" % opts.subject) / \
               ("ses-%sh%sm%ss%s" % session_groups) / "microscopy"
    name = "sub-%s_run-%s_sample-%s_stain-%s_chunk-%s_spim" % (
        opts.subject, opts.run, opts.sample, opts.stain, opts.chunk
    )
    print(str(dir_path / name))


def write_sidecar(opts):
    with open(opts.template) as fd:
        sidecar = json.load(fd)
    volume_path = pathlib.Path(opts.volume)
    ar = ArrayReader(volume_path.as_uri(), format=opts.volume_format)
    if opts.metadata_file is None:
        metadata_path = volume_path.parent / "metadata.txt"
    else:
        metadata_path = pathlib.Path(opts.metadata_file)
    x_step_size, y_voxel_size = get_sizes(metadata_path)
    sidecar["PixelSize"] = [x_step_size, y_voxel_size, x_step_size]
    sidecar["FieldOfView"] = [a * b for a, b in zip(reversed(ar.shape),
                                                    sidecar["PixelSize"])]
    sidecar["SampleStaining"] = opts.stain
    with open(opts.output, "w") as fd:
        json.dump(sidecar, fd, indent=2)

def get_xyz_from_path(key):
    z = int(key.stem)
    x, y = [int(_) for _ in key.parent.name.split("_")]
    return x, y, z

def write_transform(opts):
    dcimg_files = opts.dcimg_files
    all_paths = order_dcimg_files(dcimg_files)
    x0, y0, z0 = get_xyz_from_path(all_paths[0])
    xi, yi, zi = get_xyz_from_path(pathlib.Path(opts.input))
    xp = (xi - x0) * sqrt(2) / 10 / opts.y_voxel_size
    yp = (yi - y0) / 10 / opts.y_voxel_size
    zp = (zi - z0) * sqrt(2) / 10 / opts.y_voxel_size
    d = dict(
        SourceReferenceFrame="original",
        TargetReferenceFrame=opts.target_reference_frame,
        TransformationType="translation-3d",
        TransformationParameters=dict(
            x=xp, y=yp, z=zp)
        )
    with open(opts.output, "w") as fd:
        json.dump([d], fd, indent=2)

def order_dcimg_files_cmd(opts):
    all_paths = order_dcimg_files(opts.dcimg_files)
    print(" ".join([str(_) for _ in all_paths]))

def order_dcimg_files(dcimg_files):
    def sortfn(key: pathlib.Path):
        x, y, z = get_xyz_from_path(key)
        return x, y, z

    all_paths = sorted([pathlib.Path(_) for _ in dcimg_files],
                       key=sortfn)
    return all_paths


def get_sizes(metadata_path):
    lines = [line.strip() for line in open(metadata_path, encoding="latin1")]
    fields = lines[1].split("\t")
    x_step_size = float(fields[3])
    y_voxel_size = float(fields[2])
    return x_step_size, y_voxel_size


def rewrite_transforms(opts):
    with open(opts.align_file) as fd:
        alignment = json.load(fd)
    alignments = dict([(tuple([int(_) for _ in json.loads(k)]), v) for k, v
                       in alignment["alignments"].items()])
    for transform_filename in opts.transform_files:
        with open(transform_filename) as fd:
            transform = json.load(fd)
        x, y, z = [int(transform[0]["TransformationParameters"][_]) for _ in "xyz"]
        if (x, y, z) in alignments:
            new_x, new_y, new_z = alignments[x, y, z]
            transform[0]["TransformationParameters"]["x"] = new_x
            transform[0]["TransformationParameters"]["y"] = new_y
            transform[0]["TransformationParameters"]["z"] = new_z
            with open(transform_filename, "w") as fd:
                json.dump(transform, fd, indent=2)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    opts.func(opts)


if __name__=="__main__":
    main()