import argparse
import glymur
import multiprocessing
import pathlib
import sys
import tqdm
import typing

from ..dcimg import DCIMG


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the DCIMG file to be saved as JPEG2000")
    parser.add_argument(
        "--output-pattern",
        required=True,
        help="Output pattern for file names"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="# of worker processes"
    )
    parser.add_argument(
        "--psnr",
        default="30,40,50,60,70,80",
        help="The signal-to-noise ratios of the different levels to be "
             "created in DB. Default is 80 DB. For lossles, end with \"0\"."
    )
    return parser.parse_args(args)


dcimg:DCIMG = None


def do_one(idx:int, dest:str, psnr:typing.Sequence[int]):
    """
    convert one frame of dcimg to jpeg 2000
    :param idx: index of the frame to convert
    :param dest: path to jpeg 2000 file
    :param psnr: a sequence of DB levels for the compression
    """
    img = dcimg.read_frame(idx)
    glymur.Jp2k(dest, data=img, psnr=psnr)


def main(args=sys.argv[1:]):
    global dcimg

    opts = parse_args(args)
    paths = set()
    dcimg = DCIMG(opts.input)
    psnr = [float(_) for _ in opts.psnr.split(",")]
    with multiprocessing.Pool(opts.n_workers) as pool:
        futures = []
        for i in range(dcimg.n_frames):
            path = pathlib.Path(opts.output_pattern % i)
            parent = path.parent
            if parent not in paths:
                paths.add(parent)
                if not parent.is_dir():
                    parent.mkdir()
            futures.append(pool.apply_async(do_one, (i, str(path), psnr)))
        for future in tqdm.tqdm(futures):
            future.get()


if __name__=="__main__":
    main()