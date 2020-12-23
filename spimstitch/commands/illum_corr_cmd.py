import argparse
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.linear_model import RANSACRegressor
from ..dcimg import DCIMG
import sys
import tifffile
import tqdm


def parse_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dcimg",
        nargs="+",
        default=[],
        help="The dcimg or multiple dcimg files to be analyzed"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Name of the .tiff file holding the illumination function"
    )
    parser.add_argument(
        "--intermediate-output",
        help="Name of the .tiff file of the cumulative image before modeling. "
        "This is useful for troubleshooting"
    )
    parser.add_argument(
        "--n-frames",
        help="# of frames/planes to accumulate. Default is all of them.",
        type=int
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=1024,
        help="The number of bins for the accumulator of histograms per pixel "
    )
    parser.add_argument(
        "--values-per-bin",
        type=int,
        default=4,
        help="The number of values to be accumulated in each bin "
        "<values-per-bin> * <n-bins> is the effective maximum value to be"
        "accumulated. For instance if the maximum intensity in your images "
        "is 4095, then <n-bins>=1024 and <values-per-bin>=4 will accumulate "
        "that."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="The number of samples taken per RANSAC round."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.,
        help="At each pixel position, reduce the bins by taking this quantile "
        "of all pixels at that bin"
    )
    parser.add_argument(
        "--background",
        type=int,
        default=150,
        help="Don't consider pixels whose median value is below this."
    )
    parser.add_argument(
        "--rotate-90",
        type=int,
        default=0,
        help="Rotate image 90 degrees this many times"
    )
    parser.add_argument(
        "--flip-ud",
        action="store_true",
        help="Flip image upside down if flag is set."
    )
    return parser.parse_args(args)


def do_one_plane(dcimg:DCIMG, idx:int, opts):
    n_bins = opts.n_bins
    values_per_bin = opts.values_per_bin
    img = dcimg.read_frame(idx)
    mask = (img >= opts.background) & (img < values_per_bin * n_bins)
    img = np.clip( img // values_per_bin, 0, n_bins-1)
    img = np.rot90(img, k=opts.rotate_90)
    if opts.flip_ud:
        img = np.flipud(img)
    n_slots = np.prod(img.shape)
    return coo_matrix(
        (np.ones(np.sum(mask), np.uint32), (np.arange(n_slots)[mask.ravel()],
                                       img[mask])),
        shape=(n_slots, n_bins))


def main(args=sys.argv[1:]):
    opts = parse_arguments(args)
    dcimgs = [DCIMG(_) for _ in opts.dcimg]
    cumsum, total = build_histogram(dcimgs, opts)
    image = estimate_correction_image(cumsum, dcimgs, opts, total)
    tifffile.imsave(opts.output, image, compress=3)


def estimate_correction_image(cumsum, dcimgs, opts, total):
    qbin = np.argmin(np.abs(
        cumsum - opts.percentile * cumsum[:, :, -1, None] / 100), 2)
    if opts.intermediate_output is not None:
        intermediate = qbin * opts.values_per_bin
        tifffile.imsave(opts.intermediate_output,
                        intermediate.astype(np.uint16),
                        compress=3)
    y, x = np.mgrid[0:dcimgs[0].y_dim, 0:dcimgs[0].x_dim]
    mask = (qbin < opts.n_bins-1) & \
           (qbin >= opts.background // opts.values_per_bin)
    xa = x[mask]
    ya = y[mask]
    qbina = qbin[mask]
    model = RANSACRegressor(min_samples=opts.min_samples)
    model.fit(np.column_stack([xa, xa * xa, ya]), qbina)
    print("Image = %.4f x + %.4f x**2 + %.4f y + %.2f" %
          (model.estimator_.coef_[0], model.estimator_.coef_[1],
           model.estimator_.coef_[2], model.estimator_.intercept_))
    image = model.predict(np.column_stack([_.ravel() for _ in (x, x * x, y)])) \
        .reshape(y.shape[0], y.shape[1])
    image =  np.clip(image * opts.values_per_bin, opts.background, None)
    return image


def build_histogram(dcimgs, opts):
    total = sum([_.n_frames for _ in dcimgs])
    n_frames = min(opts.n_frames or total, total)
    dcimg_idx = np.concatenate([np.ones(dcimg.n_frames, np.uint64) * i
                                for i, dcimg in enumerate(dcimgs)])
    frame_idx = np.concatenate([np.arange(dcimg.n_frames) for dcimg in dcimgs])
    if opts.n_frames is not None:
        idxs = np.random.RandomState(1234).choice(
            total, n_frames, replace=False)
        idxs = np.sort(idxs)
        dcimg_idx = dcimg_idx[idxs]
        frame_idx = frame_idx[idxs]
    first = True
    for didx, fidx in tqdm.tqdm(zip(dcimg_idx, frame_idx),
                                total=len(dcimg_idx)):
        dcimg = dcimgs[didx]
        if first:
            accumulator = do_one_plane(dcimg, fidx, opts)
            first = False
        else:
            accumulator += do_one_plane(dcimg, fidx, opts)
    cumsum = np.cumsum(accumulator.toarray().reshape(dcimgs[0].y_dim,
                                                     dcimgs[0].x_dim,
                                                     opts.n_bins), axis=2)
    return cumsum, n_frames


if __name__ == "__main__":
    main()