import os
import glob
import logging
import cv2
import csv
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
        rec_list.append(siftObj)
        
        
    for i in rec_list:
        for j in rec_list:
            if i != j:
                num_strong_matches = match(config,i,j)
                print(num_strong_matches)


    
def write_matches(primary_image, secondary_image, strong_matches, image_destination):

    kp1 = primary_image.keypoints
    kp2 = secondary_image.keypoints

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

    matches_drawn = cv2.drawMatches(primary_image.image, kp1, secondary_image.image, kp2, strong_matches, None, **draw_params)

    image_path2 = (re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
                   "___" + re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title)))
    image_path = (image_destination + "/" + str(image_path2))

    cv2.imwrite(str(image_path),matches_drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])


def match(config, primary_sift, secondary_sift):

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(primary_sift.descriptors,secondary_sift.descriptors,k=2)

    # ratio test as per Lowe's paper
    strong_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            strong_matches.append(m)
    
    writer = csv.writer(open(config.get("results.matching_data"),'a'))
    writer.writerow([primary_sift.path,secondary_sift.path,len(strong_matches)])
        
    results_dir = config.get("results.directory")
    write_matches(primary_sift, secondary_sift, strong_matches, results_dir)
    return len(strong_matches)


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
