import configparser

import os, sys
import time, datetime
from copy import deepcopy
import threading
import argparse
import glob
import re
import traceback
from pathlib import Path, PurePath
import json

# advanced
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
import markov_clustering as mc
import networkx as nx
import math
import csv

#import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io



import classfunction
import imageProcessing
import templating
global mark_array
########################### END IMPORTS ########################################


################################################################################
def normailze_matrix(score_matrix):
    'Used to normalize the score matrix with respect to the highest value present'

    # save score matrix with number of keypoint matches for Markov
    score_matrix_copy = score_matrix

    # get max score
    max_matrix = score_matrix.max()

    # normalize
    score_matrix = score_matrix / max_matrix

    # add identity matrix
    score_matrix = score_matrix + np.identity(len(score_matrix[1]))

    return score_matrix


################################################################################
def score_boosting(primary_image, secondary_image, good_points, parameters):
    'uses image characteristics to boost scores'
    score = len(good_points)

    if (primary_image.station == secondary_image.station):
        if (primary_image.camera == secondary_image.camera):
            if (primary_image.date == secondary_image.date):
                score = score * float(parameters['score_boosting']['date_score'])
            else:
                score = score * float(parameters['score_boosting']['camera_score'])
        else:
            score = score * float(parameters['score_boosting']['station_score'])

    
    return score
################################################################################

def write_matches(kp_1, kp_2, good_points, primary_image, secondary_image, image_destination):
    'This function takes the output of the KNN matches and draws all the matching points'
    'between the two images. Writes the final product to the output directory'
    

    # parameters to pass into drawing function
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       flags = 0)


    # draw the matches between two upper pictures and horizontally concatenate
    result = cv2.drawMatches(
        primary_image.image,
        kp_1,
        secondary_image.image,
        kp_2,
        good_points,
        None,
        **draw_params) # draw connections

    # use the cv2.drawMatches function to horizontally concatenate and draw no
    # matching lines. this creates the clean bottom images.
    result_clean = cv2.drawMatches(
        primary_image.image,
        None,
        secondary_image.image,
        None,
        None,
        None) # don't draw connections
    
    # This code is Ross Pitman. I dont exactly know what all the constants are but they
    # create the border and do more image preprocessing
    row, col= result.shape[:2]
    bottom = result[row-2:row, 0:col]
    bordersize = 5
    result_border = cv2.copyMakeBorder(
        result,
        top = bordersize,
        bottom = bordersize,
        left = bordersize,
        right = bordersize,
        borderType = cv2.BORDER_CONSTANT, value = [0,0,0] )
	
    # same as above
    row, col= result_clean.shape[:2]
    bottom = result_clean[row-2:row, 0:col]
    bordersize = 5
    result_clean_border = cv2.copyMakeBorder(
        result_clean,
        top = bordersize,
        bottom = bordersize,
        left = bordersize,
        right = bordersize,
        borderType = cv2.BORDER_CONSTANT, value = [0,0,0] )
	
    # vertically concatenate the matchesDrawn and clean images created before.
    result_vertical_concat = np.concatenate(
        (result_border, result_clean_border),
        axis = 0)

    # Take the image_destination and turn it into a Path object.
    # Then add the image names to the new path.
    # # TODO: For some reason it says the 'image_destination' object is
    #           a str type at this point in the program even though it is not.
    #           Look into why.
    """image_path = image_destination.joinpath(str(len(good_points)) +
    "___" +
    re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
    "___" +
    re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title)))
    """

    image_path_helper = (re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
                   "___" + re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title)))
    image_path = (image_destination + "/" + str(len(good_points)) + "___" + str(image_path_helper))

    # Finally, write the finished image to the output folder.
    cv2.imwrite(str(image_path), result_vertical_concat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
	
################################################################################
def match(primary_images, secondary_images, image_destination,
            start_i, score_matrix, write_threshold, parameters):
    'main function used for determining matches between two images.'
    'Finds the sift keypoints/descriptors and uses a KNN based matcher'
    'to filter out bad keypoints. Writes final output to score_matrix'

    # Begin loop on the primary imags to match. Due to multithreading of the
    # program this may not be the full set of images.
    descriptors = []
    descriptors_MCL = []
    distances = []
    clustering_counter = 0
    primary_counter = 0
    count = 0
    
    for primary_count in range(len(primary_images)):
        primary_counter = primary_counter + 1
        num_desc = 0
        countDesctoPrimary = 0
        new_arr = 0
        new_arr = np.asarray(new_arr)

        print("\t\tMatching: " + os.path.basename(primary_images[primary_count].image_title) + "\n")
        # create mask from template and place over image to reduce ROI
        #mask_1 = cv2.imread(primary_images[primary_count].template_title, -1) 
        #mySift = cv2.xfeatures2d.SIFT_create()
        #kp_1, desc_1 = mySift.detectAndCompute(primary_images[primary_count].image, mask_1)
        
        kp_1 = primary_images[primary_count].key_points
        desc_1 = primary_images[primary_count].kp_desc
        
        for i in desc_1:
            descriptors.append(i)

        # paramter setup and create nearest nieghbor matcher
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        count = count + 1

        # Begin nested loopfor the images to be matched to. This secondary loop
        # will always iterate over the full dataset of images.
        for secondary_count in range(len(secondary_images)):
            countDesctoPrimary = countDesctoPrimary + 1

            # check if same image; if not, go into sophisticated matching
            if primary_images[primary_count].image_title != secondary_images[secondary_count].image_title:

                 kp_2 = secondary_images[secondary_count].key_points
                 desc_2 = secondary_images[secondary_count].kp_desc
                 # create mask from template
                 #mask_2 = cv2.imread(secondary_images[secondary_count].template_title, -1)
                 #kp_2, desc_2 = mySift.detectAndCompute(secondary_images[secondary_count].image, mask_2)

                 # check for matches
                 try:
                     # Check for similarities between pairs
                     matches = flann.knnMatch(desc_1, desc_2, k=2)
                     

                     # Use Lowe's ratio test
                     good_points = []
                     
                     for m, n in matches:
                         if m.distance < 0.7 * n.distance:
                             good_points.append(m)
                             descriptors_MCL.append(desc_1[primary_count])

                     
                     # RANSAC
                     
                     if (int(parameters['config']['ransac'])):
                         src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_points ]).reshape(-1,1,2)
                         dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_points ]).reshape(-1,1,2)

                         # used to detect bad keypoints
                         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                         matchesMask = mask.ravel().tolist()

                         h,w = primary_images[primary_count].image.shape[1], primary_images[primary_count].image.shape[0]
                         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                         dst = cv2.perspectiveTransform(pts,M)


                     # take smallest number of keypoints between two images
                     number_keypoints = 0
                     if len(kp_1) <= len(kp_2):
                         number_keypoints = len(kp_1)
                     else:
                         number_keypoints = len(kp_2)

                     # score boosting
                     score = score_boosting(primary_images[primary_count],
                        secondary_images[secondary_count], good_points, parameters)

                     # add the number of good points to score_matrix. start_i is
                     # passed in as a paramter to ensure that the correct row of the
                     # score matrix is being written to. Give this index the number
                     # of 'good_points' from the output of the KNN matcher.
                     score_matrix[start_i + primary_count][secondary_count] = score

                     # only do image processing if number of good points
                     # exceeeds threshold
                     
                     if len(good_points) > write_threshold:
                         write_matches(kp_1, kp_2, good_points,
                            primary_images[primary_count], secondary_images[secondary_count],
                            image_destination)
                    

                 except cv2.error as e:
                     print('\n\t\tERROR: {0}\n'.format(e))
                     print("\t\tError matching: " + os.path.basename(primary_images[primary_count].image_title) +
                         " and " + os.path.basename(secondary_images[secondary_count].image_title) + "\n")
    
    return score_matrix

################################################################################
def slice_generator(
        sequence_length,
        n_blocks):
    """ Creates a generator to get start/end indexes for dividing a
        sequence_length into n blocks
    """
    return ((int(round((b - 1) * sequence_length/n_blocks)),
             int(round(b * sequence_length/n_blocks)))
            for b in range(1, n_blocks+1))


################################################################################
def match_multi(primary_images, image_destination, n_threads, write_threshold, parameters):
    'Wrapper function for the "match". This also controls the multithreading'
    'if the user has declared to use multiple threads'

    # deep copy the primary_images for secondary images
    secondary_images = primary_images

    # init score_matrix
    num_pictures = len(primary_images)
    score_matrix = np.zeros(shape = (num_pictures, num_pictures))

    # prep for multiprocessing; slices is a 2D array that specifies the
    # start and end array index for each program thread about to be created
    slices = slice_generator(num_pictures, n_threads)
    thread_list = list()

    print("\tImages to pattern match: {0}\n".format(str(num_pictures)))

    # start threading
    for i, (start_i, end_i) in enumerate(slices):

        thread = threading.Thread(target = match,
                    args = (primary_images[start_i: end_i],
                            secondary_images,
                            image_destination,
                            start_i,
                            score_matrix,
                            write_threshold,
                            parameters))
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

    return score_matrix

################################################################################

################################################################################
################## 			TESTING FUNCTIONS 				 ###################
################################################################################
def test_accuracy(rec_list, score_matrix, threshold):

    # True positive: number of times a match was correctly classified as a match
    TP = 0 
    # False negative: number of times a correct match was dismissed as not a match
    FN = 0 
    # False positive:  number of times two image were incorrectly matched
    FP = 0 
    # True negative: the number of matches that were corectly dismissed as not matches
    TN = 0 

    # traverse rows
    for row in range(score_matrix.shape[0]):
        primary_cat = rec_list[row].cat_ID
        primary_title = rec_list[row].image_title

        # traverse columns
        for column in range(score_matrix.shape[1]):
            # leave out the diagonal 
            if (row != column):
                secondary_cat = rec_list[column].cat_ID

                if (score_matrix[row][column] > threshold and primary_cat == secondary_cat):
                    TP = TP + 1
                elif (score_matrix[row][column] <= threshold and primary_cat == secondary_cat):
                    FN = FN + 1
                elif (score_matrix[row][column] > threshold and primary_cat != secondary_cat):
                    FP = FP + 1
                else:
                    TN = TN + 1

    accuracy = ((TP+FP) / (TP+FP+TN+FN)) * 100
    recall = (TP) / (TP+FN)
    specificity = (TN) / (TN+FP)
    precision = (TP) / (TP+FP)


    print("\n\n")
    print("Hits:",TP)
    print("Misses:", FN)
    print("Missclassifications:", FP)
    print("\n")
    print("Accuracy:", accuracy)
    print("Recall/Sensitivity:", recall)
    print("Specificity:", specificity)
    print("Precision:", precision)
