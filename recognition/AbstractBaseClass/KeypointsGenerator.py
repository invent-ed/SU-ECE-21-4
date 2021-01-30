from abc import ABCMeta, abstractmethod


class KeypointsGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_and_save_keypoints(self, imageObj):
        pass
