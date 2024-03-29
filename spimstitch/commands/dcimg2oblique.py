import argparse
import multiprocessing
import glob
import glymur
import os
import numpy as np
from scipy.ndimage import zoom
import tifffile
from blockfs import Directory
from precomputed_tif.blockfs_stack import BlockfsStack
from precomputed_tif.ngff_stack import NGFFStack
from pystripe.core import filter_streaks, correct_lightsheet
from pystripe.core import normalize_flat, apply_flat
import sys

from spimstitch.ngff import NGFFDirectory
from ..oblique import spim_to_blockfs, get_blockfs_dims
from ..stack import SpimStack
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
        "--chunk-size",
        help="The size of a precomputed chunk in x,y,z format"
    )
    parser.add_argument(
        "--x-step-size",
        help="X step size in microns",
        type=float,
        default = 3.625 / (2 ** .5)
    )
    parser.add_argument(
        "--y-voxel-size",
        help="Y voxel size in microns",
        type=float,
        default=3.625
    )
    parser.add_argument(
        "--magnification",
        help="For a 2 camera setup, the camera with the longer arm will have an image"
             "whose magnification is less than the camera with the shorter arm. This "
             "magnification factor should be applied to stacks coming from the camera "
             "with the larger arm. The default is no magnification.",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--magnification-plane-center",
        help="The point in the plane to be magnified that corresponds to the center of"
             "the images coming from the other camera. The format is \"x,y\". The "
             "default is 1024,1024",
        default="1024,1024"
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
    parser.add_argument(
        "--jp2k",
        help="Interpret the input as a glob expression for JPEG 2000 files",
        action="store_true"
    )
    parser.add_argument(
        "--ngff",
        help="Output an NGFF volume instead of blockfs",
        action="store_true"
    )
    return parser.parse_args(args)


MY_DCIMG:DCIMG = None
MY_OPTS = None
FLAT:np.ndarray = None


def do_one_dcimg(sidx:str) -> np.ndarray:
    img = MY_DCIMG.read_frame(int(sidx))
    img = do_one(img)
    return img


def do_one_jp2000(path:str) -> np.ndarray:
    img = glymur.Jp2k(path)[:]
    return do_one(img)


def magnify(img):
    zoomed = zoom(img, MY_OPTS.magnification)
    center_x, center_y = [int(int(_) * MY_OPTS.magnification)
              for _ in MY_OPTS.magnification_plane_center.split(",")]
    x0 = center_x - img.shape[1] // 2
    x1 = center_x + img.shape[1] // 2
    y0 = center_y - img.shape[0] // 2
    y1 = center_y + img.shape[0] // 2
    return zoomed[y0:y1, x0:x1]


def do_one(img):
    img = np.rot90(img, MY_OPTS.rotate_90)
    if MY_OPTS.flip_ud:
        img = np.flipud(img)
    if MY_OPTS.magnification != 1.0:
        img = magnify(img)
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

    if MY_OPTS.jp2k:
        paths=sorted(glob.glob(MY_OPTS.input))
        fn = do_one_jp2000
        img = glymur.Jp2k(paths[0])
        x_extent = img.shape[1]
        y_extent = img.shape[0]
    else:
        MY_DCIMG = DCIMG(MY_OPTS.input)
        start = MY_OPTS.start
        stop = MY_OPTS.stop or MY_DCIMG.n_frames
        x_extent = int(MY_DCIMG.x_dim)
        y_extent = int(MY_DCIMG.y_dim)
        paths = [str(i) for i in range(start, stop)]
        fn = do_one_dcimg
    stack = SpimStack(paths, 0, 0, x_extent, y_extent, 0)
    #
    # The stack dimensions are a little elongated because of the
    # parallelogram
    #
    z_extent, y_extent, x_extent, dtype = get_blockfs_dims(
        stack, x_extent, y_extent)
    kwargs = {}
    if MY_OPTS.chunk_size is not None:
        cx, cy, cz = [int(_) for _ in MY_OPTS.chunk_size.split(",")]
        kwargs["chunk_size"] = (cz, cy, cx)
    if MY_OPTS.ngff:
        bfs_stack = NGFFStack((z_extent, y_extent, x_extent),
                              MY_OPTS.output,
                              **kwargs)
        bfs_stack.create()
    else:
        bfs_stack = BlockfsStack((z_extent, y_extent, x_extent),
                                 MY_OPTS.output, **kwargs)
    y_voxel_size = MY_OPTS.y_voxel_size
    xz_voxel_size = y_voxel_size / np.sqrt(2)
    x_step_size = MY_OPTS.x_step_size
    voxel_size = int(x_step_size * 1000),\
        int(y_voxel_size * 1000), \
        int(xz_voxel_size * 1000)
    bfs_stack.write_info_file(MY_OPTS.levels, voxel_size)
    if MY_OPTS.ngff:
        directory = NGFFDirectory(bfs_stack)
        directory.create()
    else:
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
                    voxel_size=y_voxel_size,
                    x_step_size=x_step_size,
                    read_fn=fn)
    for level in range(2, MY_OPTS.levels+1):
        bfs_stack.write_level_n(level, n_cores=MY_OPTS.n_writers)


if __name__=="__main__":
    main()