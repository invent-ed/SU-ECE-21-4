import os
import glob
from collections import namedtuple
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator


def main():

    config = JsonConfig("config.json")
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config, maskGenerator)

    Recognition = namedtuple("Recognition", ["image", "sift"])
    rec_list = []

    for image_path in list_of_images(config):
        print("PROCESSING:", image_path)
        imageObj = SnowLeopardImage(image_path)
        siftObj = keypointsGenerator.generate_keypoints_if_not_exist(imageObj)
        rec_list.append(Recognition(imageObj, siftObj))

    rec_list[3]


def list_of_images(config):
    images_dir = config.get("images.directory")
    image_ext = config.get("images.file_extension")
    path_list = list(glob.iglob(os.path.abspath(images_dir + "/*" + image_ext)))
    return [x.replace("\\", "/") for x in path_list]


if __name__ == "__main__":
    main()
