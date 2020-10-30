import numpy as np

class DCIMG:

    HEADER_SIZE = 728
    PRE_HEADER_SIZE_V1 = 100
    PRE_HEADER_SIZE_V2 = 500
    COOKIE = b"DCIMG\0\0\0"
    #
    # The pre-header, I believe, gets written at the end of the run or maybe updated during the
    # course of the run. It has the file size and the # of planes in it.
    #
    PRE_OFFSET_N_FRAMES = 6
    PRE_OFFSET_HEADER_SIZE = 7
    PRE_OFFSET_FILESIZE = 9
    #
    # The header is an array of uint32s. These are offsets into
    # the array of uint32s
    #
    OFFSET_N_FRAMES = 15
    OFFSET_N_BYTES_PER_PIXEL = 16
    OFFSET_N_CHANNELS = 17
    OFFSET_X_DIM = 18
    OFFSET_Y_DIM = 19
    OFFSET_RASTER_LENGTH = 20
    #
    # There is a 32-bit image header in front of every image
    #
    IMG_HEADER_SIZE_V1=32

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fd:
            cookie = fd.read(len(DCIMG.COOKIE))
            if cookie != DCIMG.COOKIE:
                raise ValueError("%s is not a DCIMG file" % path)
            version_buffer = fd.read(4)
            version = np.frombuffer(version_buffer, ">u4")
            if version == 2:
                pre_header = fd.read(DCIMG.PRE_HEADER_SIZE_V2)
                ipre_header = np.frombuffer(pre_header, np.uint32)
                self.img_header_size = ipre_header[DCIMG.PRE_OFFSET_HEADER_SIZE]
            else:
                pre_header = fd.read(DCIMG.PRE_HEADER_SIZE_V1)
                self.img_header_size = DCIMG.IMG_HEADER_SIZE_V1
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
                     idx * (self.img_header_size + size)
            fd.seek(offset)
            img_header = fd.read(self.img_header_size)
            img = np.frombuffer(fd.read(size), "u%d" % self.bytes_per_pixel)
            return img.reshape(self.y_dim, self.x_dim)
