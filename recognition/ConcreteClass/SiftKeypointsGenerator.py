import os
import cv2
import logging
from AbstractBaseClass.KeypointsGenerator import KeypointsGenerator
from ConcreteClass.SiftKeypoints import SiftKeypoints


class SiftKeypointsGenerator(KeypointsGenerator):

    def __init__(self, config, maskGenerator=None):
        logging.info("Initializing keypoint generator")
        self.config = config
        self.maskGenerator = maskGenerator
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.create_sift_dir_if_not_exist()

    def create_sift_dir_if_not_exist(self):
        logging.info("Creating Sift directory")
        sift_dir = self.config.get("Keypoints.directory")
        if not os.path.exists(sift_dir):
            os.makedirs(sift_dir)

    def generate_keypoints_if_not_exist(self, imageObj):
        logging.info("Generating sift keypoints")
        maskObj = self.get_mask_if_mask_generator_exists(imageObj)
        kps_path = self.generate_kps_path(imageObj.filename)
        if not os.path.isfile(kps_path):
            keypoints, descriptors = self.generate_keypoints(imageObj, maskObj)
            SiftKeypoints.save_keypoints_to_file(kps_path, keypoints, descriptors)
        return SiftKeypoints(kps_path, maskObj)

    def get_mask_if_mask_generator_exists(self, imageObj):
        logging.info("Getting mask in the keypoint generator.")
        if self.maskGenerator is not None:
            maskObj = self.maskGenerator.generate_mask_if_not_exist(imageObj)
        else:
            maskObj = None
        return maskObj

    def generate_kps_path(self, filename):
        logging.info("Generating kp path.")
        kp_dir = self.config.get("Keypoints.directory")
        kp_ext = self.config.get("Keypoints.file_extension")
        return os.path.abspath(kp_dir).replace("\\", "/") + "/" + filename + kp_ext

    def generate_keypoints(self, imageObj, maskObj=None):
        logging.info("Generating keypoints.")
        if maskObj is not None:
            keypoints, descriptors = self.sift.detectAndCompute(imageObj.image, maskObj.mask)
        else:
            keypoints, descriptors = self.sift.detectAndCompute(imageObj.image)
        return keypoints, descriptors
