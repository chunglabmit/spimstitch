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
import sys

from precomputed_tif.client import ArrayReader

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
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
    subparser = subparsers.add_parser("get-x-step-size")
    subparser.set_defaults(func=get_x_step_size)
    subparser.add_argument(
        "metadata-file",
        help="Path to the metadata.txt file",
        required=True
    )
    subparser = subparsers.add_parser("get-y-voxel-size")
    subparser.set_defaults(func=get_y_voxel_size)
    subparser.add_argument(
        "metadata-file",
        help="Path to the metadata.txt file",
        required=True
    )
    return parser.parse_args(args)


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
        print("%d did not match a date" % source_path.name,
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


def get_sizes(metadata_path):
    lines = [line.strip() for line in open(metadata_path, encoding="latin1")]
    fields = lines[1].split("\t")
    x_step_size = float(fields[3])
    y_voxel_size = float(fields[2])
    return x_step_size, y_voxel_size


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    opts.func(opts)


if __name__=="__main__":
    main()