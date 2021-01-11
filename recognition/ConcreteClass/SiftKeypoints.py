import cv2
import pickle
from AbstractBaseClass.Keypoints import Keypoints


class SiftKeypoints(Keypoints):

    def __init__(self, kps_path, maskObj=None):
        self.path = kps_path
        self.keypoints = []
        self.descriptors = []
        self.maskObj = maskObj
        self.load_keypoints_from_file(kps_path)

    @staticmethod
    def save_keypoints_to_file(kps_path, kps, descs):
        kps_and_descs_list = []
        for kp, desc in zip(kps, descs):
            kp_and_desc = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
            kps_and_descs_list.append(kp_and_desc)
        pickle_file = open(kps_path, "wb")
        pickle.dump(kps_and_descs_list, pickle_file)
        pickle_file.close()
        del kps_and_descs_list[:]

    def load_keypoints_from_file(self, kps_path):
        pickle_file = open(kps_path, "rb")
        kps_and_descs_list = pickle.load(pickle_file)
        for kp_and_desc in kps_and_descs_list:
            [pt, size, angle, response, octave, class_id, desc] = kp_and_desc
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle, _response=response, _octave=octave, _class_id=class_id)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            del kp_and_desc
        pickle_file.close()
