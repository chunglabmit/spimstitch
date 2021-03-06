import argparse
import multiprocessing
import numpy as np
import os
import tqdm
import tifffile
import sys
from ..dcimg import DCIMG

def parse_args(args = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="DCIMG file to read",
        required=True)
    parser.add_argument(
        "--output-pattern",
        help="Output pattern for filename, e.g. \"/path/to/img_%%04d.tiff\".",
        required=True)
    parser.add_argument(
        "--compression",
        help="Compression level for tiff files",
        default=3,
        type=int)
    parser.add_argument(
        "--n-workers",
        help="# of worker processes to run",
        default=1,
        type=int)
    parser.add_argument(
        "--rotate-90",
        help="Number of times to rotate each image by 90°",
        default=3,
        type=int)
    parser.add_argument(
        "--flip-ud",
        help="Flip in the Y direction after rotating",
        action="store_true")
    parser.add_argument(
        "--start",
        help="Index of first frame to decode",
        default=0,
        type=int
    )
    parser.add_argument(
        "--stop",
        help="Index of last+1 frame to decode. Default is to the end.",
        type=int
    )
    return parser.parse_args(args)


def write_one(path, output, idx, compression, rotate, flip_ud):
    dcimg = DCIMG(path)
    img = dcimg.read_frame(idx)
    img = np.rot90(img, k=rotate)
    if flip_ud:
        img = np.flipud(img)
    tifffile.imsave(output, img, compress=compression)


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    dcimg = DCIMG(opts.input)
    output_dir = os.path.dirname(opts.output_pattern)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with multiprocessing.Pool(opts.n_workers) as pool:
        futures = []
        start = opts.start
        stop = opts.stop or dcimg.n_frames
        for i in range(start, stop):
            output = opts.output_pattern % i
            futures.append(pool.apply_async(
                write_one, (opts.input, output, i,
                            opts.compression, opts.rotate_90, opts.flip_ud)))
        for future in tqdm.tqdm(futures):
            future.get()


if __name__ == "__main__":
    main()