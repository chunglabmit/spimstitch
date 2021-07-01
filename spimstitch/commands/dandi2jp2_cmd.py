import argparse
import itertools
import multiprocessing
import pathlib
import sys

import glymur
import numpy as np
import tqdm
import zarr
from mp_shared_memory import SharedMemory

from precomputed_tif.client import DANDIArrayReader

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="""
        dandi2jp2 converts a channel as processed by the dandi-pipeline.sh
        into a stack of JPEG 2000 files, performing any necessary stitching.
        """
    )
    parser.add_argument(
        "inputs",
        help="The .ngff files that make up the stack.",
        default=[],
        nargs="*"
    )
    parser.add_argument(
        "--output-pattern",
        help="The output pattern for JPEG 2000 files, e.g. "
             "\"/path-to/img_%04d.jp2\".",
        required=True
    )
    parser.add_argument(
        "--memory",
        help="Amount of memory to be taken up in reading the volume, in GB",
        default=256,
        type=float
    )
    parser.add_argument(
        "--chunk-x",
        help="Size of partial chunk acquired when reading into memory",
        default=128,
        type=int
    )
    parser.add_argument(
        "--chunk-y",
        help="Size of partial chunk acquired when reading into memory",
        default=128,
        type=int
    )
    parser.add_argument(
        "--chunk-z",
        help="Size of partial chunk in the z direction. This may be superceded "
             "by --memory",
        default=64,
        type=int
    )
    parser.add_argument(
        "--psnr",
        help="The acceptable signal to noise ratio loss during compression",
        default=0,
        type=float
    )
    parser.add_argument(
        "--n-workers",
        help="# of worker processes to use during multiprocessing",
        default=multiprocessing.cpu_count(),
        type=int
    )
    return parser.parse_args(args)


READER:DANDIArrayReader = None
MEMORY = None


def read_block(x0, x1, y0, y1, z0, z1):
    with MEMORY.txn() as memory:
        memory[:z1 - z0, y0:y1, x0:x1] = READER[z0:z1, y0:y1, x0:x1]


def write_plane(path, z, psnr):
    with MEMORY.txn() as memory:
        glymur.Jp2k(path, data=memory[z], psnr=[psnr])


def main(args=sys.argv[1:]):
    global READER, MEMORY
    opts = parse_args(args)
    urls = [pathlib.Path(path).as_uri() for path in opts.inputs]
    READER = DANDIArrayReader(urls)
    dtype = np.dtype(READER.dtype)
    z_extent = READER.shape[0]
    y_extent = READER.shape[1]
    x_extent = READER.shape[2]
    n_z = min(opts.chunk_z, opts.memory * 1000 * 1000 * 1000 //
              y_extent // x_extent // dtype.itemsize)
    MEMORY=SharedMemory((n_z, y_extent, x_extent), dtype=dtype)
    xs = np.arange(0, x_extent, opts.chunk_x)
    ys = np.arange(0, y_extent, opts.chunk_y)
    xe = np.minimum(x_extent, xs + opts.chunk_x)
    ye = np.minimum(y_extent, ys + opts.chunk_y)
    with multiprocessing.Pool(opts.n_workers) as pool:
        for z0 in range(0, z_extent, n_z):
            z1 = min(z0 + n_z, z_extent)
            futures = []
            for (x0, x1), (y0, y1) in \
                    itertools.product(zip(xs, xe), zip(ys, ye)):

                futures.append(pool.apply_async(
                    read_block, (x0, x1, y0, y1, z0, z1)))
            for future in tqdm.tqdm(futures,
                                    desc="z=%d:%d of %d" % (z0, z1, z_extent)):
                future.get()
            futures = []
            for z in range(0, z1-z0):
                path = opts.output_pattern % (z0 + z)
                futures.append(pool.apply_async(write_plane, (path, z, opts.psnr)))
            for future in tqdm.tqdm(futures):
                future.get()


if __name__=="__main__":
    main()
