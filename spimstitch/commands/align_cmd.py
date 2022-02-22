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
from ..utils import weighted_median
from ..imaris import parse_terastitcher
from .dandi_metadata import get_chunk_transform_offsets

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Root directory of unaligned precomputed volumes"
    )
    parser.add_argument(
        "--pattern",
        help="If present, the glob pattern for the volume files"
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
    parser.add_argument(
        "--ngff",
        help="Use NGFF stacks instead of blockfs",
        action="store_true")
    parser.add_argument(
        "--imaris",
        help="Use Imaris files instead of blockfs",
        action="store_true"
    )
    parser.add_argument(
        "--terastitcher-xml",
        help="Terastitcher input file giving layout of volumes"
    )
    parser.add_argument(
        "--channel",
        help="The one-based index of the channel to use for alignment for "
        "Imaris file alignment",
        type=int,
        default=1
    )
    parser.add_argument(
        "--negative-y",
        help="Use this switch if the Y direction as read from file names "
        "is in the opposite direction with respect to Y scans",
        action="store_true"
    )
    return parser.parse_args(args)

KEY_T = typing.Tuple[int, int, int]
DKEY_T = typing.Tuple[KEY_T, KEY_T]

def find_blobs(
        key:KEY_T,
        x0s:int, x1s:int, y0s:int, y1s:int, z0s:int, z1s:int) -> \
        typing.Tuple[np.ndarray, np.ndarray]:
    """
    Find blobs in a volume

    :param key: Key to fetch the volume in question
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
    if len(probs) < opts.sample_count:
        idxs = np.arange(len(probs))
    else:
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
    if opts.pattern is None:
        pattern = "**"
    else:
        pattern = opts.pattern
    if opts.imaris:
        all_volumes = []
        for k, v in parse_terastitcher(opts.terastitcher_xml).items():
            VOLUMES[k] = v
            all_volumes.append(v)
            v.directory.current_channel = opts.channel - 1
    else:
        if opts.ngff:
            paths = [path.parent for path in
                     sorted(pathlib.Path(opts.input).glob(f"{pattern}/.zgroup"))]
        else:
            paths = sorted(pathlib.Path(opts.input)
                       .glob(f"{pattern}/1_1_1/precomputed.blockfs"))
        if len(paths) == 0:
            print("There are no precomputed.blockfs files in the path, %s" %
                  opts.input)
            sys.exit(-1)
        xum = opts.voxel_size / np.sqrt(2)
        yum = opts.voxel_size
        if opts.is_oblique:
            zum = opts.voxel_size / np.sqrt(2)
        else:
            zum = opts.x_step_size
        all_volumes = []
        for path in paths:
            if opts.ngff:
                # The NGFF files should have matching sidecar json
                # This file has the putative offset of each stack
                #
                sidecar_path = path.parent / (path.stem + ".json")
                if not sidecar_path.exists():
                    raise FileNotFoundError(
                        "%s does not have a matching sidecar file" % str(path))
                with open(sidecar_path) as fd:
                    sidecar = json.load(fd)
                    z, y, x = get_chunk_transform_offsets(sidecar)
                    x, y, z = x * xum, y * yum, z*zum
                y_sign = -1 if opts.negative_y else 1
                volume = VOLUMES[x, y, z] = StitchSrcVolume(
                    str(path),
                    opts.x_step_size,
                    opts.voxel_size,
                    z0=z,
                    is_oblique=opts.is_oblique,
                    is_ngff=True,
                    x0=x, y0=y * y_sign)
            else:
                zpath = path.parent.parent
                try:
                    z = float(zpath.name) / 10
                except ValueError:
                    z = 0
                x, y = [float(_) / 10 for _ in zpath.parent.name.split("_")]
                if opts.negative_y:
                    real_y = -y
                else:
                    real_y = y
                volume = VOLUMES[x, y, z] = StitchSrcVolume(
                    str(path), opts.x_step_size, opts.voxel_size,
                    x0=x, y0=real_y, z0=z,
                    is_oblique=opts.is_oblique)
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
        for ka in sorted(VOLUMES):
            xa, ya, za = ka
            volume_a = VOLUMES[ka]
            for kb in sorted(VOLUMES):
                xb, yb, zb = kb
                volume_b = VOLUMES[kb]
                n_different = sum([a != b for a, b in (
                    (xa, xb), (ya, yb), (za, zb))])
                if n_different != 1:
                    continue
                if (kb, ka) in overlaps:
                    continue
                if not volume_a.does_overlap(
                    volume_b.x0_global, volume_b.x1_global,
                    volume_b.y0_global, volume_b.y1_global,
                    volume_b.z0_global, volume_b.z1_global
                ):
                    continue
                contexts.extend(
                    calculate_alignments(ka, kb, pool, opts))
                overlaps.add((ka, kb))

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



def weighted_median_from_overlaps(
        dd:typing.Sequence[typing.Dict[str,float]], ka, kb) -> float:
    """
    Calculate the weighted median overlap for a particular direction

    :param dd: a sequence of dictionary results of alignment matches
    :param ka: the key for the fixed coordinate, e.g. "xa"
    :param kb: the key for the moving coordinate, e.g. "xb"
    :return: the weighted median offset from expected, using 1/(1 - correlation)
    as the weighting.
    """
    data = [d[kb] - d[ka] for d in dd]
    if len(data) == 0:
        return 0
    weights = [ 1/(1 - d["corr"] + np.finfo(np.float32).eps) for d in dd]
    return weighted_median(data, weights)


ROLLUP_T = typing.Dict[DKEY_T, typing.Dict[str, float]]


def rollup_offsets(overlaps:typing.Dict[DKEY_T, typing.Sequence[typing.Dict]],
                   xum:float, yum:float, zum:float) \
    -> typing.Tuple[ROLLUP_T, ROLLUP_T, ROLLUP_T]:
    """
    Compute the weighted median of each sequence of discovered overlaps.
    These are separated into 3 dictionaries, the X overlapping pairs,
    the Y overlapping pairs and the Z overlapping pairs.

    :param overlaps: the discovered overlaps for each overlapping region
    :param xum: X voxel size in microns
    :param yum: Y voxel size in microns
    :param zum: Z voxel size in microns
    :return: three dictionaries containing the median offsets for each
    overlapping pair.
    """
    all_x, all_y, all_z = get_all_xyz()
    rollups_x = {}
    for xa, xb in zip(all_x[:-1], all_x[1:]):
        for y in all_y:
            for z in all_z:
                rollups_x[(xa, y, z), (xb, y, z)] = \
                    dict(x_off=0, y_off=0, z_off=0)
    rollups_y = {}
    for ya, yb in zip(all_y[:-1], all_y[1:]):
        for x in all_x:
            for z in all_z:
                rollups_y[(x, ya, z), (x, yb, z)] = \
                    dict(x_off=0, y_off=0, z_off=0)
    rollups_z = {}
    for za, zb in zip(all_z[:-1], all_z[1:]):
        for x in all_x:
            for y in all_y:
                rollups_z[(x, y, za), (x, y, zb)] = \
                    dict(x_off=0, y_off=0, z_off=0)
    for key, value in overlaps.items():
        key_a, key_b = key
        assert all([av <= bv for av, bv in zip(key_a, key_b)]), \
            f"First key {key_a} should be before second {key_b}"
        if key_a[0] != key_b[0]:
            tgt = rollups_x
        elif key_a[1] != key_b[1]:
            tgt = rollups_y
        else:
            tgt = rollups_z
        x_off, y_off, z_off = [
            weighted_median_from_overlaps(value, ka, kb) * um
            for ka, kb, um in (("xa", "xb", xum),
                               ("ya", "yb", yum),
                               ("za", "zb", zum))]
        tgt[key] = dict(x_off=x_off, y_off=y_off, z_off=z_off)
    return rollups_x, rollups_y, rollups_z


def compute_new_alignment(alignments, ka, kb, rollups, negative_y):
    ny_mult = -1 if negative_y else 1
    aa = alignments[ka]
    x = aa[0] + (kb[0] - ka[0]) + rollups[ka, kb]["x_off"]
    y = aa[1] + (kb[1] - ka[1]) * ny_mult + rollups[ka, kb]["y_off"]
    z = aa[2] + (kb[2] - ka[2]) + rollups[ka, kb]["z_off"]
    alignments[kb] = (x, y, z)


def get_all_xyz():
    all_x = set()
    all_y = set()
    all_z = set()
    for k in VOLUMES.keys():
        for i, s in enumerate((all_x, all_y, all_z)):
            s.add(k[i])
    return [list(sorted(s)) for s in (all_x, all_y, all_z)]


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
    all_x, all_y, all_z = get_all_xyz()
    yum = opts.voxel_size
    xum = opts.x_step_size
    zum = yum / np.sqrt(2)
    json_overlaps = dict(voxel_sizes=[xum, yum, zum])
    for key, value in overlaps.items():
        skey = json.dumps(key)
        json_overlaps[skey] = value

    rollups_x, rollups_y, rollups_z = rollup_offsets(overlaps, xum, yum, zum)
    alignments = {
        (all_x[0], all_y[0], all_z[0]): (all_x[0], all_y[0], all_z[0])
    }
    #
    # Fill in the edges
    #
    for xa, xb in zip(all_x[:-1], all_x[1:]):
        y = all_y[0]
        z = all_z[0]
        ka = (xa, y, z)
        kb = (xb, y, z)
        compute_new_alignment(alignments, ka, kb, rollups_x, opts.negative_y)

    for ya, yb in zip(all_y[:-1], all_y[1:]):
        x = all_x[0]
        z = all_z[0]
        ka = (x, ya, z)
        kb = (x, yb, z)
        compute_new_alignment(alignments, ka, kb, rollups_y, opts.negative_y)

    for za, zb in zip(all_z[:-1], all_z[1:]):
        x = all_x[0]
        y = all_y[0]
        ka = (x, y, za)
        kb = (x, y, zb)
        compute_new_alignment(alignments, ka, kb, rollups_z, opts.negative_y)
    # do the middles
    for ya, yb in zip(all_y[:-1], all_y[1:]):
        for x in all_x:
            for z in all_z:
                ka = x, ya, z
                kb = x, yb, z
                compute_new_alignment(alignments, ka, kb, rollups_y,
                                      opts.negative_y)
    json_overlaps["alignments"] =\
        dict([(json.dumps(k), v) for k, v in alignments.items()])
    json_overlaps["align-z"] = bool(opts.align_xz)
    return json_overlaps


if __name__=="__main__":
    main()
