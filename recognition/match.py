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
def score_boosting(primary_image, secondary_image, score, parameters):
    'uses image characteristics to boost scores'
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

def write_matches(primary_image, secondary_image, strong_matches, image_destination):

    kp1 = primary_image.key_points
    kp2 = secondary_image.key_points

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

    matches_drawn = cv2.drawMatches(primary_image.image, kp1, secondary_image.image, kp2, strong_matches, None, **draw_params)

    image_path2 = (re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
                   "___" + re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title)))
    image_path = (image_destination + "/" + str(image_path2))

    cv2.imwrite(str(image_path),matches_drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

################################################################################
def match(primary_image, secondary_image, image_destination,
            score_matrix):

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(primary_image.kp_desc,secondary_image.kp_desc,k=2)

    # ratio test as per Lowe's paper
    strong_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            strong_matches.append(m)

    write_matches(primary_image, secondary_image, strong_matches, image_destination)
    return len(strong_matches)


################################################################################
def match_multi(primary_images, image_destination, parameters):

    score_matrix = np.zeros(shape = (len(primary_images), len(primary_images)))
    for indx_primary, i in enumerate(primary_images):
        for indx_secondary, j in enumerate(primary_images):
            if i != j:
                num_strong_matches = match(i,j,image_destination, score_matrix)
                score_matrix[indx_primary][indx_secondary] = score_boosting(i, j, num_strong_matches, parameters)
            else:
                score_matrix[indx_primary][indx_secondary] = 0
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
