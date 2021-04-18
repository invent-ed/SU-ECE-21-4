from abc import ABCMeta, abstractmethod


class KeypointsGenerator(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def generate_and_save_keypoints(self, imageObj):
        pass
