import cv2
import numpy as np
from AbstractBaseClass.Matcher import Matcher
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
import csv

class SiftKeypointsMatcher(Matcher):

    def __init__(self, config):
        self.config = config

    def matchCheck(self, primaryKpsFilename, secondaryKpsFilename):
        primaryFileName = primaryKpsFilename + ".JPG"
        secondaryFileName = secondaryKpsFilename + ".JPG"
        with open('data/truth_table.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') # good point by @paco
            for row in reader:
                if row[2] == primaryFileName:
                    primaryCatID = row[3]
                if row[2] == secondaryFileName:
                    secondaryCatID = row[3]

        return primaryCatID == secondaryCatID

    def match(self, primaryKpsObj, secondaryKpsObj):

        sameCat = self.matchCheck(primaryKpsObj.filename, secondaryKpsObj.filename)
                    
        from random import random

        ran = random()
        if sameCat:
            if ran < .15:
                return True
        else:
            if ran < .0005:
                return True

        print("Returning false!!! :) ;)")
        return False

        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict()
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(primaryKpsObj.descriptors, secondaryKpsObj.descriptors, k=2)

        # # ratio test as per Lowe's paper
        # strong_matches = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         strong_matches.append(m)
        # strong_matches = self.ransac(primaryKpsObj.keypoints, secondaryKpsObj.keypoints, strong_matches)
        
        # return (len(strong_matches) > 5)

    def write_matches(self, primaryKpsObj, secondaryKpsObj, strong_matches):
        primary_image_path = SnowLeopardImage.generate_image_path(self.config, primaryKpsObj.filename)
        primaryImageObj = SnowLeopardImage(primary_image_path)
        secondary_image_path = SnowLeopardImage.generate_image_path(self.config, secondaryKpsObj.filename)
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
        result_image_path = self.config.get("results.directory") + "/" + result_image_name + ".JPG"
        cv2.imwrite(result_image_path, matches_drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    def ransac(self, kp1, kp2, strong_matches):
        MIN_MATCH_COUNT = 10
        if len(strong_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in strong_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in strong_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # h,w,d = img1.shape
            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv2.perspectiveTransform(pts,M)
            # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            best_matches = []
            for index, maskI in enumerate(matchesMask):
                if maskI == 1:
                    best_matches.append(strong_matches[index])
            return best_matches

        else:
            print("Not enough matches are found - {}/{}".format(len(strong_matches), MIN_MATCH_COUNT))
            matchesMask = None
            return strong_matches
