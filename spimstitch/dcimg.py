import numpy as np
import os

class DCIMG:

    HEADER_SIZE = 832
    COOKIE = b"DCIMG\0\0\0"
    #
    # The header is an array of 210 uint32s. These are offsets into
    # the array of uint32s
    #
    OFFSET_N_FRAMES = 7
    OFFSET_FILESIZE = 10
    OFFSET_N_FRAMES2 = 41
    OFFSET_N_BYTES_PER_PIXEL = 42
    OFFSET_N_CHANNELS = 43
    OFFSET_X_DIM = 44
    OFFSET_Y_DIM = 45
    OFFSET_RASTER_LENGTH = 46
    #
    # There is a 32-bit image header in front of every image
    #
    IMG_HEADER_SIZE=32

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fd:
            cookie = fd.read(len(DCIMG.COOKIE))
            if cookie != DCIMG.COOKIE:
                raise ValueError("%s is not a DCIMG file" % path)
            header = fd.read(DCIMG.HEADER_SIZE)
        int_header = np.frombuffer(header, np.uint32)
        self.n_frames = int_header[DCIMG.OFFSET_N_FRAMES]
        self.bytes_per_pixel = int_header[DCIMG.OFFSET_N_BYTES_PER_PIXEL]
        self.n_channels = int_header[DCIMG.OFFSET_N_CHANNELS]
        self.x_dim = int_header[DCIMG.OFFSET_X_DIM]
        self.y_dim = int_header[DCIMG.OFFSET_Y_DIM]

    def read_frame(self, idx):
        size = self.bytes_per_pixel * self.x_dim * self.y_dim
        with open(self.path, "rb") as fd:
            offset = len(DCIMG.COOKIE) + DCIMG.HEADER_SIZE + \
                     idx * (DCIMG.IMG_HEADER_SIZE + size)
            fd.seek(offset)
            img_header = fd.read(32)
            img = np.frombuffer(fd.read(size), "u%d" % self.bytes_per_pixel)
            return img.reshape(self.x_dim, self.y_dim)
