import os
import tifffile
import typing


class SpimStack:
    def __init__(self, paths, x0, y0, x1, y1, z0):
        self.paths = paths
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z0 + len(paths)


class StackFrame:
    def __init__(self, path, x0, y0, z):
        self.img = tifffile.imread(path)
        self.x0 = x0
        self.x1 = x0 + self.img.shape[1]
        self.y0 = y0
        self.y1 = y0 + self.img.shape[0]
        self.z = z


def make_stack(path, ext=".tiff", shape=(2048, 2048), z0=0) \
    ->typing.Dict[typing.Tuple[int, int], SpimStack]:
    d = {}
    for root, dirs, files in os.walk(path):
        try:
            toplevel = os.path.split(root)[-1]
            x, y = [int(_) for _ in toplevel.split("_")]
        except:
            continue
        key = (x, y)
        if key not in d:
            d[key] = []
        for filename in sorted(files):
            if not filename.endswith(ext):
                continue
            pathname = os.path.join(root, filename)
            d[key].append(pathname)
    output = {}
    ys, xs = shape
    for x, y in d:
        stack = SpimStack(d[x, y], x0=x, x1=x+xs, y0=y, y1=y+ys, z0=z0)
        output[x, y] = stack
    return output