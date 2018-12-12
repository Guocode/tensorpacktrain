import cv2
import os

from tensorpack import RNGDataFlow, ImageFromFile
from tensorpack.utils.argtools import shape2d
import numpy as np

class ImageFromDir(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, dir, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """

        self.files = [os.path.join(dir,file) for file in os.listdir(dir)]
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for f in self.files:
            im = cv2.imread(f, self.imread_mode)
            assert im is not None, f
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]
            yield [im,int(f[-5])]