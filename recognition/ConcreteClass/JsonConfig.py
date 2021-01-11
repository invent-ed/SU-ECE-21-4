from AbstractBaseClass.Config import *
import json


class JsonConfig(Config):

    def __init__(self, config_file):
        self.config = None
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file) as json_data:
            self.config = json.load(json_data)

    def get(self, config_name):
        tmp_dict = self.config
        for token in config_name.split('.'):
            tmp_dict = tmp_dict[token]
        return tmp_dict
