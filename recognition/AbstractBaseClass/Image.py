from abc import ABCMeta, abstractmethod


class Image(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, image_path):
        pass

    @abstractmethod
    def display(self):
        pass
