import numpy as np

class DCIMG:

    HEADER_SIZE = 728
    HEADER_SIZE_V1 = 832
    HEADER_SIZE_V1A = 904
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
    OFFSET_HEADER_END = 24 # file offset to first image header.
    #
    # There is an image header in front of every image
    #
    OFFSET_IMAGE_HEADER_SIZE = 31

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
            else:
                pre_header = fd.read(DCIMG.PRE_HEADER_SIZE_V1)

            header = fd.read(DCIMG.HEADER_SIZE)
            int_header = np.frombuffer(header, np.uint32)
            if version == 2:
                ipre_header = np.frombuffer(pre_header, np.uint32)
                self.img_header_size = ipre_header[DCIMG.PRE_OFFSET_HEADER_SIZE]
                self.header_end = int_header[DCIMG.OFFSET_HEADER_END]
            else:
                self.img_header_size = int_header[DCIMG.OFFSET_IMAGE_HEADER_SIZE]
                if self.img_header_size == 16:
                    # This is a hack... can't figure out where to read
                    # the size of the header.
                    self.header_end = DCIMG.HEADER_SIZE_V1A + len(DCIMG.COOKIE)
                else:
                    self.header_end = DCIMG.HEADER_SIZE_V1 + len(DCIMG.COOKIE)
        self.n_frames = int_header[DCIMG.OFFSET_N_FRAMES]
        self.bytes_per_pixel = int_header[DCIMG.OFFSET_N_BYTES_PER_PIXEL]
        self.n_channels = int_header[DCIMG.OFFSET_N_CHANNELS]
        self.x_dim = int_header[DCIMG.OFFSET_X_DIM]
        self.y_dim = int_header[DCIMG.OFFSET_Y_DIM]

    def read_frame(self, idx):
        size = self.bytes_per_pixel * self.x_dim * self.y_dim
        with open(self.path, "rb") as fd:
            offset = self.header_end + idx * (self.img_header_size + size)
            fd.seek(offset)
            img_header = fd.read(self.img_header_size)
            img = np.frombuffer(fd.read(size), "u%d" % self.bytes_per_pixel)
            return img.reshape(self.y_dim, self.x_dim)
