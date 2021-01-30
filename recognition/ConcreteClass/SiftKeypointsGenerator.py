import os
import cv2
import logging
from AbstractBaseClass.KeypointsGenerator import KeypointsGenerator
from ConcreteClass.MaskImage import MaskImage
from ConcreteClass.SiftKeypoints import SiftKeypoints


class SiftKeypointsGenerator(KeypointsGenerator):

    def __init__(self, config, maskGenerator=None):
        logging.info("Initializing keypoint generator")
        self.config = config
        self.maskGenerator = maskGenerator
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.create_kps_dir_if_not_exist()

    def create_kps_dir_if_not_exist(self):
        logging.info("Creating Sift directory")
        sift_dir = self.config.get("Keypoints.directory")
        if not os.path.exists(sift_dir):
            os.makedirs(sift_dir)

    def generate_and_save_keypoints_if_not_exist(self, imageObj):
        logging.info("Generating sift keypoints if not exist")
        kps_path = SiftKeypoints.generate_keypoints_path(self.config, imageObj.filename)
        if not os.path.isfile(kps_path):
            self.generate_and_save_keypoints(imageObj, kps_path)
        return kps_path

    def generate_and_save_keypoints(self, imageObj, kps_path=None):
        logging.info("Generating sift keypoints")
        if kps_path is None:
            kps_path = SiftKeypoints.generate_keypoints_path(self.config, imageObj.filename)
        maskObj = self.get_mask_if_mask_generator_exists(imageObj)
        keypoints, descriptors = self.compute_kps_and_desc(imageObj, maskObj)
        SiftKeypoints.save_keypoints_to_file(kps_path, keypoints, descriptors)
        return kps_path

    def get_mask_if_mask_generator_exists(self, imageObj):
        logging.info("Getting mask in the keypoint generator.")
        maskObj = None
        if self.maskGenerator is not None:
            mask_path = self.maskGenerator.generate_mask_if_not_exist(imageObj)
            maskObj = MaskImage(mask_path)
        return maskObj

    def compute_kps_and_desc(self, imageObj, maskObj=None):
        logging.info("Generating keypoints")
        if maskObj is None:
            maskObj = MaskImage()
            maskObj.create_empty_mask(imageObj.image.shape[:2])
        keypoints, descriptors = self.sift.detectAndCompute(imageObj.image, maskObj.mask)
        return keypoints, descriptors
