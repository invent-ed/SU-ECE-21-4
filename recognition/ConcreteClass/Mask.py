import cv2
import numpy as np
import matplotlib.pyplot as plt
from AbstractBaseClass.Image import Image


class Mask(Image):

    def __init__(self, mask_path):
        self.path = mask_path
        self.mask = None
        self.load_mask_from_file(mask_path)

    @staticmethod
    def collapse_color_channels(masks):
        size = np.shape(masks)
        mask = np.dot(masks, [[1]] * size[2])
        mask = np.reshape(mask, size[:2])
        return np.uint8(mask * 255)

    @staticmethod
    def save_mask_to_file(mask_path, mask):
        cv2.imwrite(mask_path, mask)

    def load_mask_from_file(self, mask_path):
        self.mask = np.uint8(cv2.imread(mask_path, cv2.IMREAD_REDUCED_GRAYSCALE_8))

    def display(self):
        plt.imshow(self.mask/255)
        plt.show()
