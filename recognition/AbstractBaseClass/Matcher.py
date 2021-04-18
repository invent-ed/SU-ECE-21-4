from abc import ABCMeta, abstractmethod


class Matcher(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def match(self, primaryObj, secondaryObj):
        pass
