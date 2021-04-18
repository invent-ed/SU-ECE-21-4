import os
import json
import glob
from AbstractBaseClass.Config import *


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

    def get_image_list(self):
        images_dir = self.get("images.directory")
        image_ext = self.get("images.file_extension")
        path_list = list(glob.iglob(os.path.abspath(images_dir + "/*" + image_ext)))
        return [x.replace("\\", "/") for x in path_list]
