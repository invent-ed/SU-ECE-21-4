import cv2
import numpy as np
from AbstractBaseClass.Matcher import Matcher
from ConcreteClass.SnowLeopardImage import SnowLeopardImage


class SiftKeypointsMatcher(Matcher):

    def __init__(self, config):
        self.config = config

    def match(self, primaryKpsObj, secondaryKpsObj):
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
        strong_matches = self.ransac(primaryKpsObj.keypoints, secondaryKpsObj.keypoints, strong_matches)

        # distance_of_matches = []
        # for i in strong_matches:
        #     distance_of_matches.append(i.distance)
        # distance_of_matches.sort()
        # average = np.average(distance_of_matches)
        # standard_deviation = np.std(distance_of_matches)

        # matching_meta_data = [primaryKpsObj.filename, secondaryKpsObj.filename, len(strong_matches), average, standard_deviation]

        # writer = csv.writer(open(config.get("results.matching_data"), 'a'))
        # writer.writerow(matching_meta_data + distance_of_matches[0:10] + distance_of_matches[-10:])

        # print("writing the matches")
        #if (len(strong_matches) > 5):
            #self.write_matches(primaryKpsObj, secondaryKpsObj, strong_matches)
        
        return (len(strong_matches) > 5)

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
