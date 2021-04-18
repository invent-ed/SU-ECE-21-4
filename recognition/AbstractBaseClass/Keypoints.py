from abc import ABCMeta, abstractmethod


class Keypoints(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, kps_path):
        self.path = None
        self.keypoints = []
        self.descriptors = []

    @staticmethod
    @abstractmethod
    def save_keypoints_to_file(kps_path, kps, descs):
        pass

    @abstractmethod
    def load_keypoints_from_file(self, kps_path):
        pass