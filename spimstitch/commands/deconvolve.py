"""Deconvolve a blockfs volume using Richardson / Lucy

The code is taken from the following page:
http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

and

https://github.com/CellProfiler/tutorial/blob/master/cellprofiler-tutorial/example2a_imageprocessing.py

"""
import argparse
import itertools
import pathlib

import tqdm
from blockfs.directory import Directory
import numpy as np
import multiprocessing
from precomputed_tif.client import ArrayReader
from precomputed_tif.blockfs_stack import BlockfsStack
from scipy import ndimage
import sys
import typing


ARRAY_READER:ArrayReader = None
DIRECTORY:Directory = None


def parse_args(args:typing.Sequence[str]=sys.argv[1:]):
    """
    Parse the command-line arguments

    :param args: argument strings
    :return: the options dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="The directory of the input precomputed volume"
    )
    parser.add_argument(
        "--input-format",
        default="blockfs",
        help="The precomputed volume input format. Default is blockfs"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The output directory for the blockfs precomputed volume"
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.09,
        help="The power of the exponent in the kernel, e**(-<power>*x)"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        help="The kernel length. This must be an odd number. The default is "
        "the nearest odd integer where the kernel value is less than .01"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="The number of iterations for the Richardson-Lucy loop",
        default=10
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=5,
        help="# of levels in the resulting precomputed volume"
    )
    parser.add_argument(
        "--voxel-size",
        default="1.8,1.8,2.0",
        help="The voxel size in microns: three comma-separated numbers. "
             "Default is 1.8,1.8,2.0"
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="# of processes used to perform the deconvolution"
    )
    parser.add_argument(
        "--n-writers",
        type=int,
        default=min(13, multiprocessing.cpu_count()),
        help="# of processes used when writing the blockfs volume"
    )
    return parser.parse_args(args)


def kernel_size(power:float, fraction:float=.01) -> int:
    """
    The size of the kernel needed for all values to be above the minimum
    attenuation fraction.

    :param power: power for the exponential in the kernel
    :param fraction: Desired minimum fraction
    :return: an odd integer which is the kernel size needed
    """
    fkernel_size = -np.log(fraction) / power
    k_size = int(np.ceil(fkernel_size) // 2 ) * 2 + 1
    return k_size


def do_one(x0:int, y0:int, z0:int, power:float, klen:int, iterations:int):
    """
    Process one block.

    :param x0: the x start of the block
    :param y0: the y start of the block
    :param z0: the z start of the block
    :param power: the power of the exponential
    :param klen: the length of the kernel
    :param iterations: # of iterations in the Richardson / Lucy loop
    """
    zs, ys, xs = DIRECTORY.get_block_size(x0, y0, z0)
    x1 = x0 + xs
    y1 = y0 + ys
    z1 = z0 + zs

    psf = np.zeros((klen, 1, klen), np.float32)
    a = np.arange(klen)
    psf[a, 0, klen-1-a] = np.exp(-np.arange(klen)*power)
    x0a = max(x0 - klen, 0)
    x1a = min(x1 + klen, ARRAY_READER.shape[2])
    y0a = y0
    y1a = y1
    z0a = max(z0 - klen, 0)
    z1a = min(z1 + klen, ARRAY_READER.shape[0])
    img = ARRAY_READER[z0a:z1a, y0a:y1a, x0a:x1a]
    minimum = np.min(img)
    scale = np.max(img) - np.min(img) + np.finfo(np.float32).eps
    observed = (img.astype(np.float32) - minimum) / scale
    latent_est = np.ones_like(observed) / 2
    psf_hat = psf[::-1, ::-1, ::-1]
    for _ in range(iterations):
        est_conv = ndimage.convolve(latent_est, psf)
        relative_blur = observed / (est_conv + np.finfo(observed.dtype).eps)
        error_est = ndimage.convolve(relative_blur, psf_hat)
        latent_est = latent_est * error_est
    output = (latent_est[z0-z0a:z1-z0a, y0-y0a:y1-y0a, x0-x0a:x1-x0a] *
        scale + minimum).astype(ARRAY_READER.dtype)
    DIRECTORY.write_block(output, x0, y0, z0)


def main(args:typing.Sequence[str]=sys.argv[1:]):
    global ARRAY_READER, DIRECTORY
    opts = parse_args(args)
    if opts.kernel_size is None:
        klen = kernel_size(opts.power)
    else:
        klen = opts.kernel_size
    voxel_size = [float(_) * 1000 for _ in opts.voxel_size.split(",")]
    ARRAY_READER = ArrayReader(pathlib.Path(opts.input).as_uri(),
                               format=opts.input_format)
    bfs_stack = BlockfsStack(ARRAY_READER.shape, opts.output)
    bfs_stack.write_info_file(opts.levels, voxel_size=voxel_size)
    bfs_level1_dir = \
        pathlib.Path(opts.output) / "1_1_1" / BlockfsStack.DIRECTORY_FILENAME
    bfs_level1_dir.parent.mkdir(parents=True, exist_ok=True)
    DIRECTORY = Directory(ARRAY_READER.shape[2],
                          ARRAY_READER.shape[1],
                          ARRAY_READER.shape[0],
                          ARRAY_READER.dtype,
                          str(bfs_level1_dir),
                          n_filenames=opts.n_writers)
    DIRECTORY.create()
    DIRECTORY.start_writer_processes()
    xr = range(0, DIRECTORY.x_extent, DIRECTORY.x_block_size)
    yr = range(0, DIRECTORY.y_extent, DIRECTORY.y_block_size)
    zr = range(0, DIRECTORY.z_extent, DIRECTORY.z_block_size)
    futures = []
    with multiprocessing.Pool(opts.n_cores) as pool:
        for x0, y0, z0 in itertools.product(xr, yr, zr):
            futures.append(
                pool.apply_async(
                    do_one,
                    (x0, y0, z0, opts.power, klen, opts.iterations)))

        for future in tqdm.tqdm(futures,
                                desc="Writing blocks"):
            future.get()
        DIRECTORY.close()
    for level in range(2, opts.levels+1):
        bfs_stack.write_level_n(level, n_cores=opts.n_writers)


if __name__ == "__main__":
    main()