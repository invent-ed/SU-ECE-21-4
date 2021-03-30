import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from AbstractBaseClass.Image import *


class SnowLeopardImage(Image):
    def __init__(self, image_path=None, cat_id=None):
        self.path = None
        self.image = None
        self.filename = None
        self.ext = None
        self.station = None
        self.camera = None
        self.date = None
        self.time = None
        self.cat_id = cat_id
        if image_path is not None:
            self.load_image_and_metadata(image_path)

    @staticmethod
    def generate_image_path(config, filename):
        logging.info("Generating image path")
        img_dir = config.get("images.directory")
        img_ext = config.get("images.file_extension")
        return os.path.abspath(img_dir).replace("\\", "/") + "/" + filename + img_ext

    @staticmethod
    def save_image_to_file(image_path, image):
        logging.info("Saving image to file")
        cv2.imwrite(image_path, image)

    def load_image_and_metadata(self, image_path):
        logging.info("Setting image path")
        self.path = image_path
        self.load_image_from_file(image_path)
        self.extract_filename_and_extension(image_path)
        #self.extract_camera_trap_info(self.filename)

    def load_image_from_file(self, image_path):
        logging.info("Loading image from file")
        self.image = np.array(cv2.imread(image_path))

    def extract_filename_and_extension(self, image_path):
        logging.info("Extracting filename and extension of image file")
        base = os.path.basename(image_path)
        self.filename = os.path.splitext(base)[0]
        self.ext = os.path.splitext(base)[1]

    def extract_camera_trap_info(self, filename):
        logging.info("Extracting camera trap info")
        camera_info = filename.split("__")
        self.station = camera_info[1]
        self.camera = camera_info[2]
        self.date = camera_info[3]
        self.time = camera_info[4]

    def display(self):
        logging.info("Displaying image")
        plt.imshow(self.image)
        plt.show()
