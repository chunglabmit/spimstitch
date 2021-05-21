import argparse
import itertools
import json
import multiprocessing
import numpy as np
import pathlib
import sys

import typing

import tifffile
import tqdm
from scipy import ndimage

from ..stitch import StitchSrcVolume


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Root directory of unaligned precomputed volumes",
        required=True
    )
    parser.add_argument(
        "--output",
        help="Name of the .json output file containing the alignment"
    )
    parser.add_argument(
        "--voxel-size",
        default=1.8,
        type=float,
        help="Size of camera voxel in microns"
    )
    parser.add_argument(
        "--x-step-size",
        default=1.28,
        type=float,
        help="Size of one x-stepper step in microns"
    )
    parser.add_argument(
        "--is-oblique",
        action="store_true",
        help="Use this flag if the volumes were generated using stack2oblique"
    )
    parser.add_argument(
        "--n-cores",
        default=min(20, multiprocessing.cpu_count()),
        type=int,
        help="# of processors to use in multiprocessing"
    )
    parser.add_argument(
        "--sigma",
        default=2.5,
        type=float,
        help="The smoothing sigma in microns"
    )
    parser.add_argument(
        "--sample-count",
        help="# of locations to sample per adjoining pair",
        default=20,
        type=int
    )
    parser.add_argument(
        "--window-size",
        help="Size of 3D overlap window in which to calculate correlation. "
        "The format is x,y,z (odd integers)",
        default="21,21,21"
    )
    parser.add_argument(
        "--blob-detection-window-size",
        help="Size of 3D windows in which to look for blobs. Format is x,y,z",
        default="64,64,64"
    )
    parser.add_argument(
        "--border-size",
        help="The size of the border to use when fetching the moving "
        "volume. Larger sizes will make fewer, but larger fetches. "
        "Smaller sizes are appropriate for smaller adjustments. "
        "Format is x,y,z",
        default="10,10,10"
    )
    parser.add_argument(
        "--min-correlation",
        help="Alignments are only accepted if their correlations are at "
        "least this much. Correlations are between -1 and 1, but good values "
        "are .9 or higher",
        type=float,
        default=.95
    )
    parser.add_argument(
        "--align-xz",
        help="Adjust X and Z alignments as well as y",
        action="store_true"
    )
    return parser.parse_args(args)


def find_blobs(
        key:typing.Tuple[int, int, int],
        x0s:int, x1s:int, y0s:int, y1s:int, z0s:int, z1s:int) -> \
        typing.Tuple[np.ndarray, np.ndarray]:
    """
    Find blobs in a volume

    :param volume: the volume in question
    :param x0s: the starting x coordinate in global coordinates
    :param x1s: the ending x coordinate
    :param y0s: the starting y
    :param y1s: ending y
    :param z0s: starting z
    :param z1s: ending z
    :return: the column-stacked coordinates and the intensity of the
    difference of gaussians at each point.
    """
    volume:StitchSrcVolume = VOLUMES[key]
    a = volume.read_block(x0s, x1s, y0s, y1s, z0s, z1s).astype(np.float32)
    sigma_low = [2 / um for um in (volume.zum, volume.yum, volume.xum)]
    sigma_high = [s*3 for s in sigma_low]
    dog = ndimage.gaussian_filter(a, sigma_low) -\
          ndimage.gaussian_filter(a, sigma_high)
    r = np.array([s for s in sigma_low])
    ir = np.ceil(r).astype(int)
    grid = np.mgrid[-ir[0]:ir[0]+1, -ir[1]:ir[1]+1, -ir[2]:ir[2]+1] / \
           r[:, None, None, None]
    footprint = np.sqrt(np.sum(np.square(grid), 0)) <= 1
    maxima = (ndimage.grey_dilation(dog, footprint=footprint) == dog) &\
             (dog > 0)
    labels, counts = ndimage.label(maxima)
    zi, yi, xi = np.where(labels > 0)
    areas = np.bincount(labels[zi, yi, xi])
    idxs = np.where(areas > 0)[0]
    z, y, x = [(np.bincount(labels[zi, yi, xi], weights=_)[idxs] / areas[idxs])
               .astype(int)
               for _ in (zi, yi, xi)]
    coords = np.column_stack([z+z0s, y+y0s, x+x0s])
    cmin = np.array([[z0s+2, y0s+2, x0s+2]])
    cmax = np.array([[z1s-3, y1s-3, x1s-3]])
    mask = np.all((coords > cmin) & (coords < cmax), 1) & (dog[z, y, x] > 0)
    return coords[mask, :], dog[z[mask], y[mask], x[mask]]


def calculate_alignments(key_a:typing.Tuple[float, float, float],
                         key_b:typing.Tuple[float, float, float],
                         pool:multiprocessing.Pool,
                         opts):
    """
    Calculate alignments for this volume, queueing them on a multiprocessing
    pool

    :param key_a: the key into VOLUMES for the first of the
                  overlapping volumes (fixed)
    :param key_b: the key into VOLUMES for the second of the
                  overlapping volumes (moving)
    :param pool: do the actual calculations on this pool
    :param opts: the options for making the alignments
    :return: a sequence of futures for each alignment.
    """
    volume_a:StitchSrcVolume = VOLUMES[key_a]
    volume_b:StitchSrcVolume = VOLUMES[key_b]
    #
    # Random state for choosing points to investigate
    #
    seed = (np.array((key_a, key_b)) * 10).astype(np.int64).flatten() \
           % (2 ** 31)
    r = np.random.RandomState(seed)

    hw_x, hw_y, hw_z = [int(_)//2 for _ in opts.window_size.split(",")]
    bw_x, bw_y, bw_z = [int(_) for _ in opts.border_size.split(",")]
    sigma_x = opts.sigma / volume_a.xum
    sigma_y = opts.sigma / volume_a.yum
    sigma_z = opts.sigma / volume_a.zum
    #
    # The dimensions of the intersection of the volumes
    #
    (z0i, y0i, x0i), (z1i, y1i, x1i) = volume_a.find_overlap(volume_b)
    #
    # The volume to search
    #
    x0s = x0i + hw_x + bw_x
    x1s = x1i - hw_x - bw_x
    y0s = y0i + hw_y + bw_y
    y1s = y1i - hw_y - bw_y
    z0s = z0i + hw_z + bw_z
    z1s = z1i - hw_z - bw_z
    if x1s <= x0s or y1s <= y0s or z1s <= z0s:
        return []

    #
    # This is a rough blob-finder, not designed to be comprehensive,
    # just get some of them.
    #
    hbw_x, hbw_y, hbw_z = \
        [int(_) // 2  for _ in opts.blob_detection_window_size.split(",")]
    #
    # Come up with a few random patches in which to look for blobs
    #
    x0bw = x0s + hbw_x
    x1bw = max(x1s - hbw_x, x0bw + 1)
    y0bw = y0s + hbw_y
    y1bw = max(y1s - hbw_y, y0bw + 1)
    z0bw = z0s + hbw_z
    z1bw = max(z1s + hbw_z, z0bw + 1)
    xbs = r.randint(x0bw, x1bw, opts.sample_count)
    ybs = r.randint(y0bw, y1bw, opts.sample_count)
    zbs = r.randint(z0bw, z1bw, opts.sample_count)
    futures = []
    xbs0 = np.maximum(x0s, xbs - hbw_x)
    xbs1 = np.minimum(x1s, xbs + hbw_x)
    ybs0 = np.maximum(y0s, ybs - hbw_y)
    ybs1 = np.minimum(y1s, ybs + hbw_y)
    zbs0 = np.maximum(z0s, zbs - hbw_z)
    zbs1 = np.minimum(z1s, zbs + hbw_z)
    for (x0sa, x1sa, y0sa, y1sa, z0sa, z1sa) in \
            zip(xbs0, xbs1, ybs0, ybs1, zbs0, zbs1):
        futures.append(pool.apply_async(
            find_blobs,
            (key_a, x0sa, x1sa, y0sa, y1sa, z0sa, z1sa)))
    points = []
    dogs = []
    for future in futures:
        p, d = future.get()
        points.append(p)
        dogs.append(d)

    points = np.concatenate(points, 0)
    dogs = np.concatenate(dogs)
    probs = dogs / np.sum(dogs)
    idxs = r.choice(len(probs), opts.sample_count, replace=False, p=probs)
    xs = points[idxs, 2]
    ys = points[idxs, 1]
    zs = points[idxs, 0]
    contexts = []
    first = True
    for xa, ya, za in zip(xs, ys, zs):
        context = dict(key_a=key_a,
                       key_b=key_b,
                       xa=int(xa), ya=int(ya), za=int(za),
                       future=pool.apply_async(
                           do_one_align,
                           (key_a, key_b, xa, ya, za,
                            (hw_z, hw_y, hw_x),
                            (sigma_z, sigma_y, sigma_x),
                            (bw_z, bw_y, bw_x))
                       ))
        contexts.append(context)
    return contexts


def do_one_align(key_a, key_b, x, y, z, pad, sigma, border):
    volume_a:StitchSrcVolume = VOLUMES[key_a]
    volume_b:StitchSrcVolume = VOLUMES[key_b]
    result = volume_a.align(volume_b, x, y, z, pad, sigma, border)
    if result[0] > .95 and False:
        hp = pad[2] // 2
        filename = "/media/disk2/data/snapshots/%04d_%04d_%04d_before.tiff" % (x, y, z)
        img = np.zeros((hp*2+1, hp*2+1, 3), np.uint16)
        img[:, :, 0] = volume_a.read_block(x-hp, x+hp+1, y-hp, y+hp+1, z, z+1)[0]
        img[:, :, 1] = volume_b.read_block(x-hp, x+hp+1, y-hp, y+hp+1, z, z+1)[0]
        tifffile.imsave(filename, img)
        filename = "/media/disk2/data/snapshots/%04d_%04d_%04d_after.tiff" % (x, y, z)
        corr, (zm, ym, xm) = result
        img = np.zeros((hp*2+1, hp*2+1, 3), np.uint16)
        img[:, :, 0] = volume_a.read_block(x-hp, x+hp+1, y-hp, y+hp+1, z, z+1)[0]
        img[:, :, 1] = volume_b.read_block(xm-hp, xm+hp+1, ym-hp, ym+hp+1, zm, zm+1)[0]
        tifffile.imsave(filename, img)
    return result

VOLUMES={}


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    #
    # Collect the stacks. These are in the format
    # opts.input/X/X_Y/Z/1_1_1/precomputed.blockfs
    #
    paths = sorted(pathlib.Path(opts.input)
                   .glob("**/1_1_1/precomputed.blockfs"))
    if len(paths) == 0:
        print("There are no precomputed.blockfs files in the path, %s" %
              opts.input)
        sys.exit(-1)
    xum = opts.voxel_size / np.sqrt(2)
    yum = opts.voxel_size
    if opts.is_oblique:
        zum = opts.x_step_size / np.sqrt(2)
    else:
        zum = opts.x_step_size
    all_volumes = []
    for path in paths:
        zpath = path.parent.parent
        try:
            z = float(zpath.name) / 10
        except ValueError:
            z = 0
        x, y = [float(_) / 10 for _ in zpath.parent.name.split("_")]
        volume = VOLUMES[x, y, z] = StitchSrcVolume(str(path),
                                                    opts.x_step_size,
                                                    opts.voxel_size,
                                                    z, opts.is_oblique)
        if not opts.is_oblique:
            volume.x0 = z
            volume.xum = xum
            volume.y0 = y
            volume.yum = yum
            volume.z0 = x - z
            volume.zum = zum
        all_volumes.append(volume)
    StitchSrcVolume.rebase_all(all_volumes, z_too=True)

    contexts = []
    overlaps = set()
    with multiprocessing.Pool(opts.n_cores) as pool:
        for (xa, ya, za), volume_a in VOLUMES.items():
            for (xb, yb, zb), volume_b in VOLUMES.items():
                n_different = sum([a != b for a, b in (
                    (xa, xb), (ya, yb), (za, zb))])
                if n_different != 1:
                    continue
                if ((xb, yb, zb), (xa, ya, za)) in overlaps:
                    continue
                if not volume_a.does_overlap(
                    volume_b.x0_global, volume_b.x1_global,
                    volume_b.y0_global, volume_b.y1_global,
                    volume_b.z0_global, volume_b.z1_global
                ):
                    continue
                contexts.extend(
                    calculate_alignments((xa, ya, za), (xb, yb, zb), pool, opts))
                overlaps.add(((xa, ya, za), (xb, yb, zb)))

        overlaps = {}
        for context in tqdm.tqdm(contexts):
            corr, (zb, yb, xb) = context["future"].get()
            del context["future"]
            if corr < opts.min_correlation:
                continue
            context["xb"], context["yb"], context["zb"] =\
                [int(_) for _ in (xb, yb, zb)]
            context["corr"] = float(corr)
            key = (context["key_a"], context["key_b"])
            if key not in overlaps:
                overlaps[key] = []
            overlaps[key].append(context)
    json_overlaps = make_json_alignment_dict(overlaps, opts)
    with open(opts.output, "w") as fd:
        json.dump(json_overlaps, fd, indent=2)


def make_json_alignment_dict(overlaps:dict, opts):
    """
    Make a json-serializable dictionary for output. The dictionary structure:

    voxel_size - the best guess at the correct voxel size, if we deem the
                 putative voxel size to be at fault

    voxel_sizes - the voxel sizes calculated from each of the overlaps

    alignments - a dictionary. For each block whose alignment needs to be
                 corrected, the key is the JSON serializable coordinates
                 of the block in microns, as given by the file names. The
                 value is the x, y, z triplet of the calculated true position
                 of the block in microns.

    other keys are JSON-serializable pairs of triplets giving the two blocks
    with overlaps. The values of each of these are sequences of dictionaries
    with keys of:

        xa, ya, za - the global coordinate of a point in the first block
        xb, yb, zb - the global coordinate of the point as found in the
                     second block
        corr - the final Pearson cross-correlation between patches at the
        two coordinates.
    :param overlaps: A dictionary of overlaps, similar in structure to the
                     one serialized above.
    :param opts: the command-line options
    :return: a JSON-serializable dictionary as described above.
    """
    json_overlaps = {}
    est_voxel_sizes = []
    um_y_offsets = {}
    um_x_offsets = {}
    um_z_offsets = {}
    xz_pix_size = opts.voxel_size / np.sqrt(2)
    for key, value in overlaps.items():
        skey = json.dumps(key)
        json_overlaps[skey] = value
        key_a, key_b = key
        volume_a = VOLUMES[key_a]
        volume_b = VOLUMES[key_b]
        for d in value:
            if key_a[2] != key_b[2]:
                xa = d["xa"]
                xb = d["xb"]
                xpix = volume_a.x_relative(0) - volume_b.x_relative(0) + xb - xa
                xdist = volume_b.x0 - volume_a.x0
                xum = xdist / xpix
                est_voxel_sizes.append(xum * np.sqrt(2))
            elif key_a[1] != key_b[1]:
                ya = d["ya"]
                yb = d["yb"]
                ypix = volume_a.y_relative(0) - volume_b.y_relative(0) + yb - ya
                ydist = volume_b.y0 - volume_a.y0
                yum = ydist / ypix
                est_voxel_sizes.append(yum)
                um_y_offset = float((ya - yb) * opts.voxel_size)
                um_x_offset = float((d["xa"] - d["xb"]) * xz_pix_size)
                um_z_offset = float((d["za"] - d["zb"]) * xz_pix_size)
                if key_b[1] > key_a[1]:
                    keyy = key_b[1]
                else:
                    keyy = key_a[1]
                    um_x_offset = -um_x_offset
                    um_y_offset = -um_y_offset
                    um_z_offset = -um_z_offset
                xz_key = (key_a[0], key_a[2])
                for d, v in ((um_x_offsets, um_x_offset),
                             (um_y_offsets, um_y_offset),
                             (um_z_offsets, um_z_offset)):
                    if xz_key not in d:
                        d[xz_key] = {}
                    if keyy not in d[xz_key]:
                        d[xz_key][keyy] = []
                    d[xz_key][keyy].append(v)
    json_overlaps["voxel_size"] = np.median(est_voxel_sizes)
    json_overlaps["voxel_sizes"] = [float(_) for _ in est_voxel_sizes]
    if len(um_y_offsets) > 0:
        alignments = {}
        for xz_key in um_y_offsets:
            for y_key in um_y_offsets[xz_key]:
                json_key = json.dumps((xz_key[0], y_key, xz_key[1]))
                alignments[json_key] = [xz_key[0], y_key, xz_key[1]]
        for xz_key in um_y_offsets:
            for i, offsets in enumerate((
                    um_x_offsets, um_y_offsets, um_z_offsets)):
                if (not opts.align_xz) and i != 1:
                    continue
                accumulator = 0
                for y_key in sorted(offsets[xz_key]):
                    key = [xz_key[0], y_key, xz_key[1]]
                    offset = np.median(offsets[xz_key][y_key])
                    accumulator += offset
                    new_value = float(key[i] + accumulator)
                    json_key = json.dumps(key)
                    alignments[json_key][i] = new_value
        json_overlaps["alignments"] = alignments
    if opts.align_xz:
        json_overlaps["align-z"] = True
    return json_overlaps


if __name__=="__main__":
    main()
