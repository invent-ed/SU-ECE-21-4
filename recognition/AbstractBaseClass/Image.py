from abc import ABCMeta, abstractmethod


class Image(metaclass=ABCMeta):

    @abstractmethod
    def display(self):
        pass
