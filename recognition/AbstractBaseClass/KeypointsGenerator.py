from abc import ABCMeta, abstractmethod


class KeypointsGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_keypoints(self, imageObj):
        pass
