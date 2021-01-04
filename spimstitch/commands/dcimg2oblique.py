import argparse
import enum
import multiprocessing
import os

import numpy as np
import tifffile
from blockfs import Directory
from precomputed_tif.blockfs_stack import BlockfsStack
from pystripe.core import filter_streaks, correct_lightsheet
from pystripe.core import normalize_flat, apply_flat
import sys
from ..oblique import spim_to_blockfs, get_blockfs_dims
from ..stack import SpimStack, StackFrame
from ..dcimg import DCIMG


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Name of the DCIMG to convert to oblique",
        required=True
    )
    parser.add_argument(
        "--output",
        help="The path to the precomputed output volume",
        required=True
    )
    parser.add_argument(
        "--start",
        help="Starting frame of DCIMG for partial read. Default is to"
             "read from the beginnning",
        type=int,
        default=0
    )
    parser.add_argument(
        "--stop",
        help="Ending frame of DCIMG for partial read. Default is to read"
             "to the end",
        type=int
    )
    parser.add_argument(
        "--destripe-method",
        help="Either \"lightsheet\" to use the lightsheet method or "
             "\"wavelet\" to use the wavelet method. Default is don't "
             "destripe"
    )
    parser.add_argument(
        "--levels",
        help="Number of decimation levels for the precomputed volume",
        default=5,
        type=int
    )
    parser.add_argument(
        "--n-workers",
        help="Number of worker processes to use during reading and destriping.",
        default=multiprocessing.cpu_count(),
        type=int
    )
    parser.add_argument(
        "--n-writers",
        help="Number of writer processes to use writing the blockfs storage",
        default=min(12, multiprocessing.cpu_count()),
        type=int
    )
    # Pystripe arguments
    parser.add_argument(
        "--sigma1",
        "-s1",
        help="Foreground bandwidth [pixels], larger = more filtering",
        type=float,
        default=0)
    parser.add_argument(
        "--sigma2",
        "-s2",
        help="Background bandwidth [pixels] (Default: 0, off)",
        type=float,
        default=0)
    parser.add_argument(
        "--decomposition-levels",
        "-l",
        help="Number of decomposition levels (Default: max possible)",
        type=int, default=0)
    parser.add_argument(
        "--wavelet",
        "-w",
        help="Name of the mother wavelet (Default: Daubechies 3 tap)",
        type=str,
        default='db3')
    parser.add_argument(
        "--threshold",
        "-t",
        help="Global threshold value (Default: -1, Otsu)",
        type=float,
        default=-1)
    parser.add_argument(
        "--crossover",
        "-x",
        help="Intensity range to switch between foreground and background "
             "(Default: 10)",
        type=float,
        default=10)
    parser.add_argument(
        "--chunks",
        help="Chunk size for batch processing (Default: 1)",
        type=int,
        default=1)
    parser.add_argument(
        "--flat",
        "-f",
        help="Flat reference TIFF image of illumination pattern used for "
             "correction",
        type=str,
        default=None)
    parser.add_argument(
        "--dark",
        "-d",
        help="Intensity of dark offset in flat-field correction",
        type=float,
        default=0)
    parser.add_argument(
        "--rotate-90",
        "-r",
        help="Number of 90 degree rotations for image",
        type=int,
        default=0)
    parser.add_argument(
        "--flip-ud",
        help="Flip image in the Y direction if present",
        action="store_true")
    parser.add_argument(
        "--artifact-length",
        help="Look for minimum in lightsheet direction over this length",
        default=150,
        type=int)
    parser.add_argument(
        "--background-window-size",
        help="Size of window in x and y for background estimation",
        default=200,
        type=int)
    parser.add_argument(
        "--percentile",
        help="The percentile at which to measure the background",
        type=float,
        default=.25)
    parser.add_argument(
        "--lightsheet-vs-background",
        help="The background is multiplied by this weight when comparing "
             "lightsheet against background",
        type=float,
        default=2.0)
    return parser.parse_args(args)


MY_DCIMG:DCIMG = None
MY_OPTS = None
FLAT:np.ndarray = None

def do_one(sidx:str) -> np.ndarray:
    img = MY_DCIMG.read_frame(int(sidx))
    img = np.rot90(img, MY_OPTS.rotate_90)
    if MY_OPTS.flip_ud:
        img = np.flipud(img)
    if MY_OPTS.destripe_method == "lightsheet":
        fimg = correct_lightsheet(
            img.reshape(img.shape[0], img.shape[1], 1),
            percentile=MY_OPTS.percentile,
            lightsheet=dict(selem=(1, MY_OPTS.artifact_length, 1)),
            background=dict(
                selem=(MY_OPTS.background_window_size,
                       MY_OPTS.background_window_size, 1),
                spacing=(25, 25, 1),
                interpolate=1,
                dtype=np.float32,
                step=(2, 2, 1)),
            lightsheet_vs_background=MY_OPTS.lightsheet_vs_background
        ).reshape(img.shape[0], img.shape[1])
    elif MY_OPTS.destripe_method == "wavelet":
        sigma = [MY_OPTS.sigma1, MY_OPTS.sigma2]
        fimg = filter_streaks(
            img, sigma,
            level=MY_OPTS.decomposition_levels,
            wavelet=MY_OPTS.wavelet,
            crossover=MY_OPTS.crossover,
            threshold=MY_OPTS.threshold,
            flat=FLAT,
            dark=MY_OPTS.dark)
    else:
        fimg = img.astype(np.float32)
    if FLAT is not None:
        img = apply_flat(fimg, FLAT)
    img = np.clip(img, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return img


def main(args=sys.argv[1:]):
    global MY_OPTS, MY_DCIMG, FLAT
    MY_OPTS = parse_args(args)
    destripe_method = MY_OPTS.destripe_method
    if not (destripe_method is None or
            destripe_method == "lightsheet" or
            destripe_method == "wavelet"):
        print(
            "--destripe-method must be \"lightsheet\", \"wavelet\" or blank",
            file=sys.stderr)
        sys.exit(-1)

    if MY_OPTS.flat is not None:
        FLAT = normalize_flat(tifffile.imread(MY_OPTS.flat))

    MY_DCIMG = DCIMG(MY_OPTS.input)
    start = MY_OPTS.start
    stop = MY_OPTS.stop or MY_DCIMG.n_frames
    x_extent = int(MY_DCIMG.x_dim)
    y_extent = int(MY_DCIMG.y_dim)
    z_extent = int(stop - start)
    paths = [str(i) for i in range(start, stop)]
    stack = SpimStack(paths, 0, 0, x_extent, y_extent, 0)
    #
    # The stack dimensions are a little elongated because of the
    # parallelogram
    #
    z_extent, y_extent, x_extent, dtype = get_blockfs_dims(
        stack, x_extent, y_extent)
    bfs_stack = BlockfsStack((z_extent, y_extent, x_extent),
                             MY_OPTS.output)
    bfs_stack.write_info_file(MY_OPTS.levels)
    bfs_level1_dir = os.path.join(
        MY_OPTS.output, "1_1_1", BlockfsStack.DIRECTORY_FILENAME)
    if not os.path.exists(os.path.dirname(bfs_level1_dir)):
        os.mkdir(os.path.dirname(bfs_level1_dir))
    directory = Directory(x_extent,
                          y_extent,
                          z_extent,
                          np.uint16,
                          bfs_level1_dir,
                          n_filenames=MY_OPTS.n_writers)
    directory.create()
    directory.start_writer_processes()
    spim_to_blockfs(stack, directory, MY_OPTS.n_workers,
                    read_fn=do_one)
    for level in range(2, MY_OPTS.levels+1):
        bfs_stack.write_level_n(level, n_cores=MY_OPTS.n_writers)


if __name__=="__main__":
    main()