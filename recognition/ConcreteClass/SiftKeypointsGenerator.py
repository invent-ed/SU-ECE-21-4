import os
import cv2
import numpy as np
from AbstractBaseClass.KeypointsGenerator import KeypointsGenerator
from ConcreteClass.Mask import Mask
from ConcreteClass.SiftKeypoints import SiftKeypoints


class SiftKeypointsGenerator(KeypointsGenerator):

    def __init__(self, config, maskGenerator=None):
        self.config = config
        self.maskGenerator = maskGenerator
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.create_sift_dir_if_not_exist()

    def create_sift_dir_if_not_exist(self):
        sift_dir = self.config.get("Keypoints.directory")
        if not os.path.exists(sift_dir):
            os.makedirs(sift_dir)

    def generate_keypoints_if_not_exist(self, imageObj):
        mask_path = self.maskGenerator.generate_mask_path(imageObj.filename)
        kps_path = self.generate_kps_path(imageObj.filename)
        if not os.path.isfile(mask_path) or not os.path.isfile(kps_path):
            keypoints, descriptors, maskObj = self.generate_keypoints(imageObj)
            SiftKeypoints.save_keypoints_to_file(kps_path, keypoints, descriptors)
        maskObj = Mask(mask_path) if self.maskGenerator is not None else None
        return SiftKeypoints(kps_path, maskObj)

    def generate_kps_path(self, filename):
        kp_dir = self.config.get("Keypoints.directory")
        kp_ext = self.config.get("Keypoints.file_extension")
        return os.path.abspath(kp_dir).replace("\\", "/") + "/" + filename + kp_ext

    def generate_keypoints(self, imageObj):
        if self.maskGenerator is None:
            maskObj = None
            keypoints, descriptors = self.sift.detectAndCompute(imageObj.image)
        else:
            maskObj = self.maskGenerator.generate_mask_if_not_exist(imageObj)
            keypoints, descriptors = self.sift.detectAndCompute(imageObj.image, maskObj.mask)
        return keypoints, descriptors, maskObj
