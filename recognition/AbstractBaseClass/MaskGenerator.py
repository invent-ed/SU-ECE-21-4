from abc import ABCMeta, abstractmethod


class MaskGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_and_save_mask(self, imageObj):
        return
