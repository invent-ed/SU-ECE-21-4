import cv2
import numpy as np
import matplotlib.pyplot as plt
from AbstractBaseClass.Image import *


class SnowLeopardImage(Image):
    def __init__(self, image_path, cat_id=None):
        self.path = None
        self.image = None
        self.filename = None
        self.ext = None
        self.station = None
        self.camera = None
        self.date = None
        self.time = None
        self.cat_id = cat_id
        self.set_path(image_path)

    @staticmethod
    def save_image_to_file(image_path, image):
        cv2.imwrite(image_path, image)

    def set_path(self, image_path):
        self.path = image_path
        self.load_image_from_file(image_path)
        self.extract_filename_and_extension(image_path)
        self.extract_camera_trap_info(self.filename)

    def load_image_from_file(self, image_path):
        self.image = np.array(cv2.imread(image_path))

    def extract_filename_and_extension(self, image_path):
        filename_and_ext = image_path.split("/")[-1]
        self.filename = filename_and_ext[:-4]
        self.ext = filename_and_ext[-4:]

    def extract_camera_trap_info(self, filename):
        camera_info = filename.split("__")
        self.station = camera_info[1]
        self.camera = camera_info[2]
        self.date = camera_info[3]
        self.time = camera_info[4]

    def display(self):
        plt.imshow(self.image)
        plt.show()
