import os
import glob
import logging
from time import localtime, strftime
from collections import namedtuple
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator


def main():

    setup_logger()

    config = JsonConfig("data/config.json")
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config, maskGenerator)

    Recognition = namedtuple("Recognition", ["image", "sift"])
    rec_list = []

    for image_path in list_of_images(config):
        print("PROCESSING IMAGE:", image_path)
        logging.info("PROCESSING IMAGE: " + image_path)
        imageObj = SnowLeopardImage(image_path)
        siftObj = keypointsGenerator.generate_keypoints_if_not_exist(imageObj)
        rec_list.append(Recognition(imageObj, siftObj))


def setup_logger():
    FORMAT = "[%(filename)s:%(lineno)s - $(funcName)40s() ] %(message)s"
    FILENAME = "data/logs/log_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".log"
    logging.basicConfig(format=FORMAT, filename=FILENAME, level=logging.DEBUG)


def list_of_images(config):
    images_dir = config.get("images.directory")
    image_ext = config.get("images.file_extension")
    path_list = list(glob.iglob(os.path.abspath(images_dir + "/*" + image_ext)))
    return [x.replace("\\", "/") for x in path_list]


if __name__ == "__main__":
    main()
