import argparse
import contextlib
import itertools
import json
import multiprocessing
import resource
import pathlib
import sys
import uuid

import typing

import hdf5plugin
import h5py
import numpy as np
import scipy
from ..dcimg import DCIMG
from .dandi_metadata import target_file_fn
from ..oblique import make_resources, PlaneR, um2oblique, get_blockfs_dims
from ..pipeline import Dependent, Pipeline
from ..stack import SpimStack

DCIMG_DICT:typing.Dict[str, DCIMG] = {}


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="A program to convert oblique DCIMG into a NGFF-like HDF5"
    )
    parser.add_argument(
        "dcimg_files",
        help="One or more DCIMG files to be processed. Each will be run on "
        "a separate process.",
        nargs="+"
    )
    parser.add_argument(
        "--dest",
        help="The root of the DANDI tree",
        required=True
    )
    parser.add_argument(
        "--y-voxel-size",
        help="The Y voxel size. This program does not support x-step-size "
             "other than Y voxel size / sqrt(2)",
        required=True
    )
    parser.add_argument(
        "--subject",
        help="The DANDI subject",
        required=True
    )
    parser.add_argument(
        "--sample",
        help="The DANDI sample ID",
        required=True
    )
    parser.add_argument(
        "--stain",
        help="The DANDI stain for all of the DCIMG files",
        required=True
    )
    parser.add_argument(
        "--n-cores",
        help="# of processes to use",
        type=int,
        default=min(multiprocessing.cpu_count(), 24)
    )
    parser.add_argument(
        "--sidecar-template",
        help="JSON sidecar template for this channel",
        required=True
    )
    return parser.parse_args(args)


def get_coords(path):
    z = int(path.stem)
    x, y = map(int, path.parent.name.split("_"))
    return x // 10, y // 10, z // 10


def dcimg_sort_key(d):
    return d["x"], d["y"], d["z"]


def dcimg_read_fn(path:str) -> np.ndarray:
    """
    Read a frame from a DCIMG file

    :param path: this is in the form of "key:frome" where key is a key into
    the DCIMG_DICT and frame is the frame number
    :return: a 2-d frame read from the DCIMG file
    """
    key, sframe = path.split(":")
    frame = int(sframe)
    return DCIMG_DICT[key].read_frame(frame)

@contextlib.contextmanager
def open_h5datasets(path:pathlib.Path, shape) \
        -> typing.Tuple[h5py.Dataset, h5py.Dataset]:
    """
    Return two datasets, the original size and decimated by 2x
    """
    multiscales = [
        dict(datasets=[dict(path="0"), dict(path="1")],
             metadata=dict(method="scipy.ndimage.map_coordinates",
                           version=scipy.__version__),
             name=path.stem,
             type="reduce",
             version="0.2")
    ]
    omero = dict(
        channels=[{
            "active": True,
            "coefficient": 1,
            "color": "FFFFFF",
            "family": "linear",
            "inverted": False,
            "label": path.stem,
            "window": {"max": 65535, "min": 0}}],
        name=path.stem,
        rdefs=dict(defaultT=0,
                   defaultZ=1024,
                   model="greyscale"),
        version="0.2"
    )
    with h5py.File(str(path), "w") as f:
        f.attrs["multiscales"] = json.dumps(multiscales)
        f.attrs["omero"] = json.dumps(omero)
        ds1 = f.create_dataset(
            "0", shape=[1, 1, shape[0], shape[1], shape[2]],
            dtype=np.uint16, chunks=(1, 1, 64, 64, 64),
            **hdf5plugin.Blosc(cname="zstd"))
        ds2 = f.create_dataset(
            "1", shape=[1, 1, shape[0] // 2, shape[1] // 2, shape[2] // 2],
            dtype=np.uint16,
            chunks=(1, 1, 64, 64, 64),
            **hdf5plugin.Blosc(cname="zstd"))
        yield ds1, ds2

class HDF5BlockD:
    def __init__(self,
                 ds1:h5py.Dataset,
                 ds2:h5py.Dataset,
                 planes:typing.Sequence[PlaneR],
                 x0:int, x1:int, y0:int, y1:int, z0:int, z1:int,
                 ys:int, y_voxel_size:float):
        self.ds1 = ds1
        self.ds2 = ds2
        self.planes = planes
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.ys = ys
        self.y_voxel_size = y_voxel_size

    def execute(self):
        block = np.zeros((self.z1 - self.z0,
                          self.y1 - self.y0,
                          self.x1 - self.x0), np.float32)
        for plane in self.planes:
            x0a = self.x0
            x1a = self.x1
            y0a = max(self.y0, 0)
            y1a = min(self.y1, plane.shape[0])

            srcy, srcx = np.mgrid[y0a:y1a, x0a - plane.z:x1a - plane.z]
            desty, destx = np.mgrid[y0a - self.y0:y1a - self.y0,
                           x0a - self.x0:x1a - self.x0]
            destz = srcx - self.z0
            mask = (destz >= 0) & (destz < block.shape[0]) & \
                   (srcx >= 0) & (srcx < plane.shape[1])
            with plane.memory.txn() as m:
                block[destz[mask], desty[mask], destx[mask]] = \
                    m[srcy[mask], srcx[mask]]
        self.ds1[0, 0, self.z0:self.z1, self.y0:self.y1, self.x0:self.x1] = block
        #
        # The decimation is the mean of a 2x2x2 block
        #
        if any([_ % 2 == 1 for _ in block.shape]):
            # If any dimension of the block has an odd dimension
            # truncate that dimension
            block = block[:block.shape[0] - (block.shape[0] % 2),
                          :block.shape[1] - (block.shape[1] % 2),
                          :block.shape[2] - (block.shape[2] % 2)]
        decimated = [block[z0::2, y0::2, x0::2].astype(np.uint32)
                     for x0, y0, z0 in itertools.product((0, 1), (0, 1), (0, 1))]
        dblock = sum(decimated[1:], decimated[0]) // len(decimated)
        self.ds2[0, 0,
                 self.z0//2:self.z1//2,
                 self.y0//2:self.y1//2,
                 self.x0//2:self.x1//2] = dblock
        del self.planes


def write_sidecar(dest_sidecar:pathlib.Path,
                  x:float,
                  y:float,
                  z:float,
                  xum:float,
                  yum:float,
                  zum:float,
                  shape:typing.Tuple[int, int, int],
                  template:typing.Dict[str, typing.Any]):
    sidecar = template.copy()
    sidecar["ChunkTransformMatrixAxis"] = ["Z", "Y", "X"]
    sidecar["PixelSize"] = [zum, yum, xum]
    sidecar["FieldOfView"] = [a * b for a, b in zip(shape,
                                                    sidecar["PixelSize"])]
    sidecar["ChunkTransformMatrix"] = [
        [zum, 0., 0., z],
        [0., yum, 0., y],
        [0., 0., xum, x],
        [0., 0., 0., 1.0]
    ]
    with dest_sidecar.open("w") as fd:
        json.dump(sidecar, fd, indent=2)

def do_one(dcimg_path:pathlib.Path,
           x:int, y:int, z:int, y_voxel_size:float,
           dest:pathlib.Path, template:typing.Dict[str, typing.Any]):
    dcimg = DCIMG(str(dcimg_path))
    xum = zum = y_voxel_size / np.sqrt(2)
    key = uuid.uuid4().hex
    DCIMG_DICT[key] = dcimg
    paths = [":".join([key, str(i)]) for i in range(dcimg.n_frames)]
    x0 = int(x / xum)
    y0 = int(y / y_voxel_size)
    stack = SpimStack(paths,
                      x0,
                      y0,
                      x0 + dcimg.x_dim,
                      y0 + dcimg.y_dim,
                      int(z / zum))
    z_extent, y_extent, x_extent, dtype = get_blockfs_dims(
        stack, dcimg.x_dim, dcimg.y_dim)
    shape = (z_extent, y_extent, x_extent)
    dest_h5 = dest.parent / (dest.name + ".h5")
    dest_sidecar = dest.parent / (dest.name + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_sidecar(dest_sidecar, x, y, z, xum, y_voxel_size, zum, shape,
                  template)
    with open_h5datasets(dest_h5, shape) as (ds1, ds2):
        pipeline = make_pipeline(ds1, ds2, stack, xum, y_voxel_size, zum, x_extent, y_extent, z_extent)
        pipeline.run_no_pool()


def make_pipeline(ds1, ds2, stack, xum, y_voxel_size, zum, x_extent, y_extent, z_extent):
    resources_and_planers = make_resources(stack,
                                           dcimg_read_fn)
    resources, planers = [[_[idx] for _ in resources_and_planers]
                          for idx in (0, 1)]
    #
    # Read 2x2x2 blocks at the original resolution so that we can
    # do the downsampled version at the same time
    #
    z_block_size, y_block_size, x_block_size = \
        [_ * 2 for _ in ds1.chunks[2:]]
    x0r = range(0, x_extent, x_block_size)
    x1r = [min(_ + x_block_size, x_extent) for _ in x0r]
    z0r = range(0, z_extent, z_block_size)
    z1r = [min(_ + z_block_size, z_extent) for _ in z0r]
    dependents = []
    for (x0, x1), (z0, z1) in itertools.product(
            zip(x0r, x1r), zip(z0r, z1r)):
        x0um = x0 * xum
        x1um = x1 * xum
        z0um = z0 * zum
        z1um = z1 * zum
        # Get the frame # of all 4 corners for minmax
        frame, _, _ = um2oblique(np.array([x0um, x1um, x0um, x1um]),
                                 np.array([0, 0, 0, 0]),
                                 np.array([z0um, z0um, z1um, z1um], ),
                                 voxel_size=y_voxel_size,
                                 x_step_size=xum)
        z0idx = max(stack.z0, int(np.min(frame)))
        z1idx = min(stack.z1, int(np.max(np.ceil(frame))))
        if z1idx <= z0idx:
            continue
        blockd = HDF5BlockD(ds1, ds2, planers[z0idx:z1idx],
                            x0, x1, 0, y_extent, z0, z1,
                            y_block_size,
                            y_voxel_size)
        dependent = Dependent(
            prerequisites=resources[z0idx:z1idx],
            fn=blockd.execute,
            name="block %d:%d %d:%d" % (x0, x1, z0, z1))
        dependents.append(dependent)
    pipeline = Pipeline(dependents)
    return pipeline


def get_args(d, ichunk, opts):
    y_voxel_size = float(opts.y_voxel_size)
    x = d["x"]
    y = d["y"]
    z = d["z"]
    dcimg_path = d["path"]
    class TFFOpts:
        subject = opts.subject
        sample = opts.sample
        run = "1"
        source_path = str(dcimg_path.parent.parent.parent.parent)
        stain = opts.stain
        chunk = str(ichunk)
    dest = pathlib.Path(opts.dest) / target_file_fn(TFFOpts())
    with open(opts.sidecar_template) as fd:
        template = json.load(fd)
    return dcimg_path, x, y, z, y_voxel_size, dest, template


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    # Hack - allow more open files.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 4096), hard))
    min_x = 1000 * 1000 * 1000
    min_y = 1000 * 1000 * 1000
    min_z = 1000 * 1000 * 1000
    all_dcimg = []
    for filename in opts.dcimg_files:
        path = pathlib.Path(filename)
        x, y, z = get_coords(path)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        all_dcimg.append(dict(x=x, y=y, z=z, path=path))
    for d in all_dcimg:
        d["x"] -= min_x
        d["y"] -= min_y
        d["z"] -= min_z
    all_dcimg = sorted(all_dcimg, key=dcimg_sort_key)
    star_args = [get_args(_, i+1, opts) for i, _ in enumerate(all_dcimg)]

    with multiprocessing.Pool(opts.n_cores) as pool:
        pool.starmap(do_one, star_args)


if __name__=="__main__":
    main()
