import os
import glob
import logging
import cv2
import csv
import numpy as np
from time import localtime, strftime
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.SiftKeypoints import SiftKeypoints
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator


def main():
    setup_logger()
    config = JsonConfig("data/config.json")
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config, maskGenerator)
 
    groups = group_by_metadata(list_of_images(config))
   	for group in groups:
   		findRepresentatives(group)



def setup_logger():
    FORMAT = "[%(filename)s:%(lineno)s - $(funcName)40s() ] %(message)s"
    FILENAME = "data/logs/log_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".log"
    logging.basicConfig(format=FORMAT, filename=FILENAME, level=logging.DEBUG)

def list_of_images(config):
    images_dir = config.get("images.directory")
    image_ext = config.get("images.file_extension")
    path_list = list(glob.iglob(os.path.abspath(images_dir + "/*" + image_ext)))
    return [x.replace("\\", "/") for x in path_list]


def filename_without_ext(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]

def find_representatives(group):
	for i, filename in enumerate(group.filenames): 
        kps_path = SiftKeypoints.generate_keypoints_path(config, filename)
        if not os.path.isfile(kps_path):
            print("GENERATING KEYPOINTS FOR IMAGE:", image_path)
            logging.info("GENERATING KEYPOINTS FOR IMAGE: " + image_path)
            imageObj = SnowLeopardImage(image_path)
            keypointsGenerator.generate_and_save_keypoints(imageObj, kps_path)
        logging.info("LOADING KEYPOINTS: " + kps_path)
        if kpsObj.length > 100: 
        	group.representative_indices.append(i)

def match_groups(groups):
	kps_list = []
	for group in groups:
	    for filename in group.filenames:
	        kps_path = SiftKeypoints.generate_keypoints_path(config, filename)
	        if not os.path.isfile(kps_path):
	            print("GENERATING KEYPOINTS FOR IMAGE:", image_path)
	            logging.info("GENERATING KEYPOINTS FOR IMAGE: " + image_path)
	            imageObj = SnowLeopardImage(image_path)
	            keypointsGenerator.generate_and_save_keypoints(imageObj, kps_path)
	        logging.info("LOADING KEYPOINTS: " + kps_path)
	        kpsObj = SiftKeypoints(kps_path)
	        if kpsObj.length > 0:
	            kps_list.append(kpsObj)

    for i, primaryKpsObj in enumerate(kps_list):
        for j, secondaryKpsObj in enumerate(kps_list):
            if i>j:
                num_strong_matches = match(config, primaryKpsObj, secondaryKpsObj)
                print("Number of strong matches: ", num_strong_matches)
                logging.info("Number of strong matches: " + str(num_strong_matches))

            #Then do some function that will group if conditions are meet

def match(config, primaryKpsObj, secondaryKpsObj):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(primaryKpsObj.descriptors, secondaryKpsObj.descriptors, k=2)

    # ratio test as per Lowe's paper
    strong_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            strong_matches.append(m)

    distance_of_matches = []
    for i in strong_matches:
    	distance_of_matches.append(i.distance)
    distance_of_matches.sort()
    average = np.average(distance_of_matches)
    standard_deviation = np.std(distance_of_matches)

    matching_meta_data = [primaryKpsObj.filename, secondaryKpsObj.filename, len(strong_matches), average, standard_deviation]

    writer = csv.writer(open(config.get("results.matching_data"), 'a'))
    writer.writerow(matching_meta_data + distance_of_matches[0:10] + distance_of_matches[-10:])
    write_matches(config, primaryKpsObj, secondaryKpsObj, strong_matches)

    return len(strong_matches)


def write_matches(config, primaryKpsObj, secondaryKpsObj, strong_matches):
    primary_image_path = SnowLeopardImage.generate_image_path(config, primaryKpsObj.filename)
    primaryImageObj = SnowLeopardImage(primary_image_path)
    secondary_image_path = SnowLeopardImage.generate_image_path(config, secondaryKpsObj.filename)
    secondaryImageObj = SnowLeopardImage(secondary_image_path)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=None,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    matches_drawn = cv2.drawMatches(
        primaryImageObj.image, primaryKpsObj.keypoints,
        secondaryImageObj.image, secondaryKpsObj.keypoints,
        strong_matches, None, **draw_params
    )

    result_image_name = primaryImageObj.filename + "___" + secondaryImageObj.filename
    result_image_path = config.get("results.directory") + "/" + result_image_name + ".JPG"
    cv2.imwrite(result_image_path, matches_drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])


# def ransac(kp1, kp2, strong_matches):
# 	MIN_MATCH_COUNT = 10
#     if len(strong_matches)>MIN_MATCH_COUNT:
#     	src_pts = np.float32([ kp1[m.queryIdx].pt for m in strong_matches ]).reshape(-1,1,2)
#     	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in strong_matches ]).reshape(-1,1,2)
#
#     	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     	matchesMask = mask.ravel().tolist()
#
#     	#h,w,d = img1.shape
#     	#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     	#dst = cv2.perspectiveTransform(pts,M)
#     	#img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
#     	best_matches = []
#     	for index, maskI in enumerate(matchesMask):
#     		if maskI == 1:
#     			best_matches.append(strong_matches[index])
#     	del strong_matches[:]
#     	strong_matches = best_matches
#
#     else:
#     	print( "Not enough matches are found - {}/{}".format(len(strong_matches), MIN_MATCH_COUNT) )
#     	matchesMask = None
#
#    	return strong_matches

if __name__ == "__main__":
    main()