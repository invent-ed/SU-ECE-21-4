from abc import ABCMeta, abstractmethod


class GroupStage(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def process(self, groups_list):
        pass


