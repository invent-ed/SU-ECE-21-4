from abc import ABCMeta, abstractmethod


class Config(metaclass=ABCMeta):

    @abstractmethod
    def get(self, config_name):
        pass
