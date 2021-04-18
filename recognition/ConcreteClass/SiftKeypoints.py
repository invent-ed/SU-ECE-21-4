import os
import cv2
import pickle
import logging
import numpy as np
from AbstractBaseClass.Keypoints import Keypoints


class SiftKeypoints(Keypoints):

    def __init__(self, kps_path, maskObj=None):
        self.path = kps_path
        self.length = None
        self.keypoints = []
        self.descriptors = []
        self.maskObj = maskObj
        self.filename = None
        self.ext = None
        if kps_path is not None:
            self.extract_filename_and_extension(kps_path)
            self.load_keypoints_from_file(kps_path)

    @staticmethod
    def generate_keypoints_path(config, filename):
        logging.info("Generating keypoints path")
        kp_dir = config.get("Keypoints.directory")
        kp_ext = config.get("Keypoints.file_extension")
        return os.path.abspath(kp_dir).replace("\\", "/") + "/" + filename + kp_ext

    @staticmethod
    def save_keypoints_to_file(kps_path, kps, descs):
        logging.info("Saving keypoints to file")
        kps_and_descs_list = []
        if descs is None:
            descs = []
        for kp, desc in zip(kps, descs):
            kp_and_desc = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
            kps_and_descs_list.append(kp_and_desc)
        pickle_file = open(kps_path, "wb")
        pickle.dump(kps_and_descs_list, pickle_file)
        pickle_file.close()
        del kps_and_descs_list[:]

    def extract_filename_and_extension(self, kps_path):
        logging.info("Extracting filename and extension of keypoints file")
        base = os.path.basename(kps_path)
        self.filename = os.path.splitext(base)[0]
        self.ext = os.path.splitext(base)[1]

    def load_keypoints_from_file(self, kps_path):
        logging.info("Loading keypoints from file")
        pickle_file = open(kps_path, "rb")
        kps_and_descs_list = pickle.load(pickle_file)
        pickle_file.close()
        for kp_and_desc in kps_and_descs_list:
            [pt, size, angle, response, octave, class_id, desc] = kp_and_desc
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle, _response=response, _octave=octave, _class_id=class_id)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            del kp_and_desc
        self.descriptors = np.asarray(self.descriptors)
        self.length = len(self.descriptors)