"""
check_dcimg_sizes.py

This is a script that should be compatible with any Python 3.0 installation
that checks all subdirectories for DCIMG files and makes sure all files
have the same number of frames.

Usage:
python3 check_dcimg_sizes.py <directory>

where <directory> is the root directory, for instance, "/path-to/Ex_488_Em_1"
"""
import pathlib
import sys

def sortkey(path:pathlib.Path):
    return tuple([int(_) for _ in path.parent.name.split("_")])

def dcimg_size(path:pathlib.Path):
    with open(path, "rb") as fd:
        fd.seek(36)
        b = fd.read(4)
    return b[0] + 256 * (b[1]  + 256 * (b[2] + 256 * b[3]))

def main():
    all_dcimg = sorted(pathlib.Path(sys.argv[1]).glob("**/*.dcimg"),
                       key=sortkey)
    if len(all_dcimg) == 0:
        print("No dcimg files along %s" % sys.argv[1])
        exit(-1)
    my_size = dcimg_size(all_dcimg[0])
    for path in all_dcimg:
        size = dcimg_size(path)
        print("%s: n_frames=%d" % (str(path), size))
        if size != my_size:
            print("%s has discrepant size" % str(path))
            exit(-1)
    print("OK")


if __name__ == "__main__":
    main()
