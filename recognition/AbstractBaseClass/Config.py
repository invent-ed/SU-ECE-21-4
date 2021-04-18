import logging
from time import localtime, strftime
from abc import ABCMeta, abstractmethod


class Config(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get(self, config_name):
        pass

    @abstractmethod
    def get_image_list(self):
        pass

    def setup_logger(self):
        FORMAT = "[%(filename)s:%(lineno)s - $(funcName)40s() ] %(message)s"
        LOG_DIR = self.get("logs.directory")
        FILENAME = LOG_DIR + "/log_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".log"
        logging.basicConfig(format=FORMAT, filename=FILENAME, level=logging.DEBUG)
