import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from AbstractBaseClass.Image import Image


class MaskImage(Image):

    def __init__(self, image_path):
        self.path = image_path
        self.mask = None
        self.filename = None
        self.ext = None
        if image_path is not None:
            self.extract_filename_and_extension(image_path)
            self.load_mask_from_file(image_path)

    @staticmethod
    def generate_mask_path(config, filename):
        logging.info("Generating mask path")
        mask_dir = config.get("Mask.directory")
        mask_ext = config.get("Mask.file_extension")
        return os.path.abspath(mask_dir).replace("\\", "/") + "/" + filename + mask_ext

    @staticmethod
    def save_mask_to_file(mask_path, mask):
        logging.info("Saving mask to file")
        cv2.imwrite(mask_path, mask)

    def extract_filename_and_extension(self, mask_path):
        logging.info("Extracting filename and extension of mask file")
        base = os.path.basename(mask_path)
        self.filename = os.path.splitext(base)[0]
        self.ext = os.path.splitext(base)[1]

    def load_mask_from_file(self, mask_path):
        logging.info("Loading mask from file")
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    def display(self):
        logging.info("Displaying mask image")
        plt.imshow(self.mask/255)
        plt.show()

    def create_empty_mask(self, mask_shape):
        self.mask = np.uint8(np.ones(mask_shape) * 255)