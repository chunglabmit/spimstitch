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
import zarr

from math import sqrt

from precomputed_tif.client import ArrayReader

MICR_DIR = "micr"
NGFF_VERSION = "0.4"

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
    build_negative_y_parser(subparsers)
    build_flip_y_parser(subparsers)
    build_machine_id_parser(subparsers)
    build_set_ngff_from_sidecar(subparsers)
    return parser.parse_args(args)

def build_set_ngff_from_sidecar(subparsers):
    subparser = subparsers.add_parser(
        "set-ngff-from-sidecar",
        description="Set the NGFF's transform and axes from the values in "
                    "the sidecar"
    )
    subparser.set_defaults(func=set_ngff_from_sidecar_opts)
    subparser.add_argument(
        "--sidecar",
        help="The path to the sidecar that has the transform set",
        required=True
    )
    subparser.add_argument(
        "--ngff",
        help="The NGFF that will have the transform in its metadata set to"
             "whatever is in the sidecar",
        required=True
    )
    subparser.add_argument(
        "--z-offset",
        help="This offset, in microns, will be added to the chunk's z-offset. "
             "For instance, for a 2mm slab, 10 from the top, this might be "
             "set to 20000",
        type=float,
        default=0.0
    )


def build_machine_id_parser(subparsers):
    subparser = subparsers.add_parser(
        "get-machine-id",
        description="Read the machine ID out of the metadata file")
    subparser.set_defaults(func=get_machine_id)
    subparser.add_argument(
        "metadata_file",
        help="Path to the metadata.txt file"
    )

def build_flip_y_parser(subparsers):
    subparser = subparsers.add_parser(
        "get-flip-y",
        description="Determine whether the given camera needs to have the "
                    "frame's Y axis flipped.")
    subparser.set_defaults(func=get_flip_y)
    subparser.add_argument(
        "metadata_file",
        help="Path to the metadata.txt file."
    )
    subparser.add_argument(
        "channel",
        help="The name of the output channel, e.g. Ex_488_Em_3. This will "
             "be parsed into numbers and the closest match to 488, 561 or"
             "642 will be used to determine whether camera 1 (488) or "
             "camera 2 (561, 642) was used."
    )

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


def build_negative_y_parser(subparsers):
    subparser = subparsers.add_parser("get-negative-y")
    subparser.set_defaults(func=get_negative_y)
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
        "--y-voxel-size",
        help="The size of a voxel in microns",
        default=1.8,
        type=float
    )
    subparser.add_argument(
        "--dcimg-input",
        help="The path to the DCIMG file being converted",
        required=True)
    subparser.add_argument(
        "--output",
        help="Name of the sidecar JSON file",
        required=True
    )
    subparser.add_argument(
        "dcimg_files",
        nargs="*",
        default=[],
        help="The remainder of the files should include all of the "
             "DCIMG files in the volume"
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
        "--y-voxel-size",
        help="The size of a voxel in microns in the image plane",
        type=float,
        required=True
    )
    subparser.add_argument(
        "sidecar_files",
        nargs="*",
        default=[],
        help="The sidecar files output by the write-sidecar subcommand"
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


def get_negative_y(opts):
    metadata_path = pathlib.Path(opts.metadata_file)
    version = get_version(metadata_path)
    lines = read_metadata(metadata_path)
    if version > (2, 0):
        the_line = lines[1]
    else:
        the_line = lines[0]
    for field in the_line.split("\t"):
        if field.startswith("NoOffset"):
            print("negative-y")
            return
    print("positive-y")


channels_and_cameras = [(488, 1), (561, 2), (642, 2)]


def get_flip_y(opts):
    metadata_path = pathlib.Path(opts.metadata_file)
    if get_version(metadata_path) > (2, 0):
        channel = opts.channel
        closest = 10000000
        closest_camera = 1
        for part in channel.split("_"):
            try:
                wavelength = int(part)
                for target, camera in channels_and_cameras:
                    score = abs(wavelength - target)
                    if score < closest:
                        closest = score
                        closest_camera = camera
            except ValueError:
                pass
        print("flip-y" if closest_camera == 1 else "do-not-flip-y")
    else:
        print("flip-y")


def get_machine_id(opts):
    metadata_path = pathlib.Path(opts.metadata_file)
    if get_version(metadata_path) > (2, 0):
        lines = read_metadata(metadata_path)
        fields = lines.split("\t")
        print(fields[-2])
    else:
        print("oSPIM1")


def read_metadata(metadata_path):
    lines = [line.strip() for line in open(metadata_path, encoding="latin1")]
    return lines


def get_version(metadata_path):
    lines = read_metadata(metadata_path)
    fields = lines[0].split("\t")
    version_field = fields[-1]
    if version_field.startswith("v"):
        try:
            return tuple([int(_) for _ in version_field[1:].split(".")])
        except ValueError:
            return (1, 0)
    return (1, 0)


def target_file(opts):
    source_path = pathlib.Path(opts.source_path)
    match = re.search("^(\\d{8})_(\\d{2})_(\\d{2})_(\\d{2})", source_path.name)
    if not match:
        print("%s did not match a date" % source_path.name,
              file=sys.stderr)
        exit(-1)
    session_groups = tuple(match.groups())
    session = "%sh%sm%ss%s" % session_groups
    dir_path = pathlib.Path("sub-%s" % opts.subject) / \
               ("ses-%s" % session) / MICR_DIR
    name = f"sub-{opts.subject}_ses-{session}_sample-{opts.sample}_" \
           f"stain-{opts.stain}_run-{opts.run}_chunk-{opts.chunk}_SPIM"
    print(str(dir_path / name))


def write_sidecar(opts):
    template_path = pathlib.Path(opts.template)
    with template_path.open() as fd:
        sidecar:dict = json.load(fd)
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
    stain_template_path = \
        template_path.parent / (template_path.stem + ".%s.json" % opts.stain)
    if stain_template_path.exists():
        with stain_template_path.open() as fd:
            sidecar.update(json.load(fd))
    dcimg_files = opts.dcimg_files
    all_paths = order_dcimg_files(dcimg_files)
    x0, y0, z0 = get_xyz_from_path(all_paths[0])
    xi, yi, zi = get_xyz_from_path(pathlib.Path(opts.dcimg_input))
    xp, yp, zp = compute_offsets(opts, x0, xi, y0, yi, z0, zi)

    set_chunk_transform_matrix(sidecar, xp, yp, zp,
                               x_step_size, y_voxel_size, x_step_size)
    with open(opts.output, "w") as fd:
        json.dump(sidecar, fd, indent=2)


def set_chunk_transform_matrix(sidecar, xp, yp, zp, xum, yum, zum):
    sidecar["ChunkTransformMatrix"] = [
        [zum, 0., 0., zp],
        [0., yum, 0., yp],
        [0., 0., xum, xp],
        [0., 0., 0., 1.0]
    ]
    sidecar["ChunkTransformMatrixAxis"] = ["Z", "Y", "X"]


def get_chunk_transform_offsets(sidecar):
    "Get offsets in z, y, x order"
    ctma = sidecar["ChunkTransformMatrixAxis"]
    idx_z = ctma.index("Z")
    idx_y = ctma.index("Y")
    idx_x = ctma.index("X")
    matrix = sidecar["ChunkTransformMatrix"]
    return matrix[idx_z][-1], matrix[idx_y][-1], matrix[idx_x][-1]


def get_xyz_from_path(key):
    z = int(key.stem)
    x, y = [int(_) for _ in key.parent.name.split("_")]
    return x, y, z


def write_transform(opts):
    dcimg_files = opts.dcimg_files
    all_paths = order_dcimg_files(dcimg_files)
    x0, y0, z0 = get_xyz_from_path(all_paths[0])
    xi, yi, zi = get_xyz_from_path(pathlib.Path(opts.input))
    xp, yp, zp = compute_offsets(opts, x0, xi, y0, yi, z0, zi)
    d = dict(
        SourceReferenceFrame="original",
        TargetReferenceFrame=opts.target_reference_frame,
        TransformationType="translation-3d",
        TransformationParameters=dict(
            XOffset=xp, YOffset=yp, ZOffset=zp)
        )
    with open(opts.output, "w") as fd:
        json.dump([d], fd, indent=2)


def compute_offsets(opts, x0, xi, y0, yi, z0, zi):
    xp = (xi - x0) * sqrt(2) / 10 / opts.y_voxel_size
    yp = (yi - y0) / 10 / opts.y_voxel_size
    zp = (zi - z0) * sqrt(2) / 10 / opts.y_voxel_size
    return xp, yp, zp


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
    if get_version((metadata_path)) < (2, 1):
        fields = lines[1].split("\t")
        if len(fields) == 4:
            x_step_size = float(fields[3])
        else:
            try:
                x_step_size = float(fields[4])
            except ValueError:
                x_step_size = float(fields[5])

        y_voxel_size = float(fields[2])
    else:
        field_names = lines[0].split("\t")
        fields = lines[1].split("\t")
        yv_idx = field_names.index("µm/pix")
        if yv_idx < 0:
            yv_idx = 2
        y_voxel_size = float(fields[yv_idx])
        xs_idx = field_names.index("X/Z Step (µm)")
        if xs_idx < 0:
            xs_idx = 3
        x_step_size = float(fields[xs_idx])
    return x_step_size, y_voxel_size


def rewrite_transforms(opts):
    with open(opts.align_file) as fd:
        alignment = json.load(fd)
    alignments = dict([(tuple([int(_) for _ in json.loads(k)]), v) for k, v
                       in alignment["alignments"].items()])
    yum = opts.y_voxel_size
    xum = zum = yum / sqrt(2)
    for sidecar_filename in opts.sidecar_files:
        sidecar_path = pathlib.Path(sidecar_filename)
        with sidecar_path.open() as fd:
            sidecar = json.load(fd)
        z, y, x = get_chunk_transform_offsets(sidecar)
        x, y, z = [int(offset * um)
                   for offset, um in
                   ((x, xum),
                    (y, yum),
                    (z, zum))]
        if (x, y, z) in alignments:
            new_x, new_y, new_z = alignments[x, y, z]
            set_chunk_transform_matrix(sidecar, new_x, new_y, new_z,
                                       xum, yum, zum)
            tmp_name = sidecar_path.parent / (sidecar_path.name + ".tmp")
            with tmp_name.open("w") as fd:
                json.dump(sidecar, fd, indent=2)
            sidecar_path.replace(tmp_name)


def set_ngff_from_sidecar_opts(opts):
    set_ngff_from_sidecar(
        pathlib.Path(opts.sidecar),
        pathlib.Path(opts.ngff),
        opts.z_offset
    )


def set_ngff_from_sidecar(sidecar_path:pathlib.Path,
                          ngff_path:pathlib.Path,
                          z_offset:float=0.0):
    """
    Set the NGFF transform from its sidecar

    :param sidecar_path: The path to the sidecar file
    :param ngff_path: The path to the NGFF file
    :param z_offset: The offset in microns of the slab, if any
    """
    with sidecar_path.open() as fd:
        sidecar = json.load(fd)
    ctma = sidecar["ChunkTransformMatrixAxis"]
    ctm = sidecar["ChunkTransformMatrix"]
    z_idx, y_idx, x_idx = [ctma.index(_) for _ in "ZYX"]
    xum, yum, zum = [ctm[_][_] for _ in (x_idx, y_idx, z_idx)]
    xoff, yoff, zoff = [ctm[_][-1] for _ in (x_idx, y_idx, z_idx)]
    zoff += z_offset
    ngff_zarr = zarr.group(zarr.NestedDirectoryStore(str(ngff_path)))
    multiscales = ngff_zarr.attrs["multiscales"]
    for scale in multiscales:
        scale["version"] = NGFF_VERSION
        scale["axes"] = [
            dict(name="t", type="time", unit="second"),
            dict(name="c", type="channel"),
            dict(name="z", type="space", unit="micrometer"),
            dict(name="y", type="space", unit="micrometer"),
            dict(name="x", type="space", unit="micrometer")
        ]
        for i, dataset in enumerate(scale["datasets"]):
            power = 2 ** i
            dataset["coordinateTransformations"] = [
                dict(type="scale",
                     scale=[1.0, 1.0, zum * power, yum * power, xum * power]),
                dict(type="translation", translation=[0, 0, zoff, yoff, xoff])
            ]
    ngff_zarr.attrs["multiscales"] = multiscales
    # Patch the OME version number too since we upgraded it
    omero = ngff_zarr.attrs["omero"]
    omero["version"] = NGFF_VERSION
    ngff_zarr.attrs["omero"] = omero


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    opts.func(opts)


if __name__=="__main__":
    main()