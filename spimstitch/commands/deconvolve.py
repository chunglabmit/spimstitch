"""Deconvolve a blockfs volume using Richardson / Lucy

The code is taken from the following page:
http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

and

https://github.com/CellProfiler/tutorial/blob/master/cellprofiler-tutorial/example2a_imageprocessing.py

"""
import argparse
import itertools
import pathlib

import torch
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
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for deconvolution."
    )
    parser.add_argument(
        "--n-blocks-per-process",
        type=int,
        default=2,
        help="# of blockfs blocks on a side to process at the same time. "
        "A larger number amortizes the padding while a smaller number "
        "conserves memory."
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


def do_one(x0:int, y0:int, z0:int, x1:int, y1:int, z1:int,
           power:float, klen:int, iterations:int, use_cpu):
    """
    Process one block.

    :param x0: the x start of the block
    :param y0: the y start of the block
    :param z0: the z start of the block
    :param power: the power of the exponential
    :param klen: the length of the kernel
    :param iterations: # of iterations in the Richardson / Lucy loop
    """

    half = klen // 2
    #
    # The convolution we choose puts the image at z=half, x=-half
    # because it is asymmetric.
    x_off = -half
    z_off = half
    x0a = max(x0 - klen, 0)
    x1a = min(x1 + klen, ARRAY_READER.shape[2])
    y0a = y0
    y1a = y1
    z0a = max(z0 - klen, 0)
    z1a = min(z1 + klen, ARRAY_READER.shape[0])
    img = ARRAY_READER[z0a:z1a, y0a:y1a, x0a:x1a]
    #
    # Make sure we have at least 1/2 of the convolution size of padding,
    # even if it is all zeros.
    #
    if z1a < z1 + half or z0a > z0 - half or \
       x1a < x1 + half or x0a > x0 - half:
        padded_img = np.zeros((z1 - z0 + klen * 2,
                               y1 - y0,
                               x1 - x0 + klen * 2))
        z0p = z0a - z0 + klen
        z1p = z0p + img.shape[0]
        x0p = x0a - x0 + klen
        x1p = x0p + img.shape[2]
        padded_img[z0p:z1p, :, x0p:x1p] = img
        z0a = z0 - klen
        z1a = z1 + klen
        x0a = x0 - klen
        x1a = x1 + klen
        img = padded_img
    minimum = np.min(img)
    scale = np.max(img) - np.min(img) + np.finfo(np.float32).eps
    observed = (img.astype(np.float32) - minimum) / scale
    latent_est = cpu_richardson_lucy(observed, klen, power, iterations)
    output = (latent_est[z0-z0a+z_off:z1-z0a+z_off,
                         y0-y0a:y1-y0a,
                         x0-x0a+x_off:x1-x0a+x_off] *
        scale + minimum).astype(ARRAY_READER.dtype)
    write_out(output, x0, x1, y0, y1, z0, z1)


def cpu_richardson_lucy(observed, klen, power, iterations):
    latent_est = np.ones_like(observed) / 2
    a = np.arange(klen)
    psf = np.zeros((klen, 1, klen), np.float32)
    psf[a, 0, klen - 1 - a] = np.exp(-np.arange(klen) * power)
    psf_hat = psf[::-1, ::-1, ::-1]
    for _ in range(iterations):
        est_conv = ndimage.convolve(latent_est, psf)
        relative_blur = observed / (est_conv + np.finfo(observed.dtype).eps)
        error_est = ndimage.convolve(relative_blur, psf_hat)
        latent_est = latent_est * error_est
    return latent_est


def write_out(output, x0, x1, y0, y1, z0, z1):
    zs, ys, xs = DIRECTORY.get_block_size(x0, y0, z0)
    xb0s = np.arange(x0, x1, xs)
    yb0s = np.arange(y0, y1, ys)
    zb0s = np.arange(z0, z1, zs)
    for xb0, yb0, zb0 in itertools.product(xb0s, yb0s, zb0s):
        zs, ys, xs = DIRECTORY.get_block_size(xb0, yb0, zb0)
        try:
            DIRECTORY.write_block(output[zb0 - z0:zb0 + zs - z0,
                                  yb0 - y0:yb0 + ys - y0,
                                  xb0 - x0:xb0 + xs - x0], xb0, yb0, zb0)
        except:
            raise

def pytorch_conv(x, k):
    ksize = len(k)
    khalfsize = ksize // 2
    acc = torch.zeros_like(x)
    for i in range(-khalfsize, khalfsize+1):
        x0s = 0
        x0d = 0
        z0s = 0
        z0d = 0
        x1s = x.shape[2]
        x1d = x.shape[2]
        z1s = x.shape[2]
        z1d = x.shape[2]
        if i < 0:
            x0s = x0s - i
            x1d = x1d + i
            z0d = z0d - i
            z1s = z1s + i
        elif i > 0:
            x0d = x0d + i
            x1s = x1s - i
            z0s = z0s + i
            z1d = z1d - i
        try:
            acc[z0d:z1d, :, x0d:x1d] = acc[z0d:z1d, :, x0d:x1d] + k[khalfsize-i] * x[z0s:z1s, :, x0s:x1s]
        except:
            print("%d:%d, %d:%d = %d:%d, %d: %d"% (z0d, z1d, x0d, x1d, z0s, z1s, x0s, x1s))
            print("Shape: %s" % str(acc[z0d:z1d, :, x0d:x1d].shape))
            raise
    return acc


def gpu_richardson_lucy(observed, klen, power, iterations):
    kernel1d = np.exp(-np.arange(klen) * power).astype(np.float32)
    inv_kernel1d = np.ascontiguousarray(kernel1d[::-1])
    latent_est = torch.ones_like(observed) / 2
    for _ in range(iterations):
        est_conv = pytorch_conv(latent_est, kernel1d)
        relative_blur = observed / (est_conv + np.finfo(np.float32).eps)
        error_est = pytorch_conv(relative_blur, inv_kernel1d)
        latent_est = latent_est * error_est
    return latent_est.cpu().numpy()


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
    nblks = opts.n_blocks_per_process
    xsize = DIRECTORY.x_block_size * nblks
    ysize = DIRECTORY.y_block_size * nblks
    zsize = DIRECTORY.z_block_size * nblks
    xr0 = np.arange(0, DIRECTORY.x_extent, xsize)
    yr0 = np.arange(0, DIRECTORY.y_extent, ysize)
    zr0 = np.arange(0, DIRECTORY.z_extent, zsize)
    xr1 = np.minimum(xr0 + xsize, DIRECTORY.x_extent)
    yr1 = np.minimum(yr0 + ysize, DIRECTORY.y_extent)
    zr1 = np.minimum(zr0 + zsize, DIRECTORY.z_extent)
    futures = []
    if opts.use_gpu:
        cores = 1
    else:
        cores = opts.n_cores
    with multiprocessing.Pool(cores) as pool:
        for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
                zip(xr0, xr1), zip(yr0, yr1), zip(zr0, zr1)):
            futures.append(
                pool.apply_async(
                    do_one,
                    (x0, y0, z0, x1, y1, z1, opts.power, klen, opts.iterations,
                     opts.use_gpu)))

        for future in tqdm.tqdm(futures,
                                desc="Writing blocks"):
            future.get()
        DIRECTORY.close()
    for level in range(2, opts.levels+1):
        bfs_stack.write_level_n(level, n_cores=opts.n_writers)


if __name__ == "__main__":
    main()