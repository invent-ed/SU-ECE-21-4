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
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import markov_clustering as mc
import networkx as nx
import math
import csv

#mask r-cnn
import random
import tensorflow as tf
import matplotlib
#import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io

# EDIT PATHS FOR MASK R-CNN LIBRARY FOLDER ################################################################################
# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
#print(ROOT_DIR)
#ROOT_DIR = "Mask_RCNN-master"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# from samples.snow_leopard import snow_leopard

#%matplotlib inline 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

global mark_array
########################### END IMPORTS ########################################

# CLASS DEFINITION
################################################################################
class Recognition:
    'This class holds an image-template pair.'
    'Keeps pairs secure and together thorughout whole process and'
    'cuts down on code bloat. Also holds the image title split into its'
    'base characterisitics and the proper cat ID'
    def __init__(self):
        self.image_title = ""
        self.image = ""
        self.template_title = ""
        self.template = ""
        self.station = ""
        self.camera = ""
        self.date = ""
        self.time = ""
        self.cat_ID = ""
        self.key_points = None
        self.kp_desc = None

    def add_image(self, image_title, image):
        self.image_title = image_title
        self.image = image

    def add_template(self, template_title, template):
        self.template_title = template_title
        self.template = template

    def add_title_chars(self, station, camera, date, time):
        self.station = station
        self.camera = camera
        self.date = date
        self.time = time

    def add_cat_ID(self, cat):
        self.cat_ID = cat
        
    def calculate_kp(self):
        mask_1 = cv2.imread(self.template_title, -1) 
        mySift = cv2.xfeatures2d.SIFT_create()
        self.key_points, self.kp_desc = mySift.detectAndCompute(self.image, mask_1)
########################### END CLASS DEFINITION ###############################

# FUNCTION DEFINITIONS (In Reverse Order of Call)

##inputs should be score_matrix, k_range
def kmeans_score_matrix(score_matrixCSV):

    # score matrix file path
    filename = score_matrixCSV

    # initializing the titles and rows list 
    fields = [] 
    rows = [] 
    cluster_arr = []
    rowsCount = 0
    # reading csv file 
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
        
        # extracting field names through first row 
        fields = next(csvreader)
        rowsCount = 1 
        for k in fields:
            cluster_arr.append(k)
    
        # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 
            rowsCount += 1
    
        
    for i in rows:
        for j in i:
            cluster_arr.append(j)

    print(len(cluster_arr))

    #preprocess array for clustering input
    divided_arr = np.array_split(cluster_arr, csvreader.line_num)
    clustering_in = np.vstack(divided_arr)
    clustering_in = np.float32(clustering_in)

    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']
    distortion = []
    clusterCount = 0

    #range for number of clusters 
    K = range(1,10)

    #clustering descriptors and using the elbow method to determine k
    for k in K:
        try:
            kmeanModel = KMeans(n_clusters=k).fit(clustering_in)
            kmeanModel.fit(clustering_in)
            distortion.append(sum(np.min(cdist(clustering_in, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / clustering_in.shape[0])
            clusterCount = clusterCount + 1
        except:
            pass


    K = range(1, clusterCount+1)
    #plt.plot(K, distortion, 'bx-')
    #plt.xlabel('k')
    #plt.ylabel('Distortion')
    #plt.title('The Elbow Method showing the optimal k (Score Matrix)')
    #plt.show()


    print('\n')
    print("done clustering\n")

def kmeans_clustering(descriptors_list, k_range): 

    print("Starting clustering....\n")

    max_decrease_index = 0
    max_decrease = 0.0

    #preprocess array of descriptors for clustering input
    new_arr = np.asarray(descriptors_list)
    print(len(new_arr))
    divided_array = np.array_split(new_arr, 128)
    clustering_input = np.vstack(divided_array)
    clustering_input = np.float32(clustering_input)

    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']
    distortions = []

    #range for number of clusters 
    K = range(1,k_range)

    #clustering descriptors and using the elbow method to determine k
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(clustering_input)
        kmeanModel.fit(clustering_input)
        distortions.append(sum(np.min(cdist(clustering_input, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / clustering_input.shape[0])

    print("Preparing plot...")

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k (descriptors)')
    plt.show()

    print('\n')
    print("done clustering\n")


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


################################################################################

################################################################################
def check_matrix(rec_list, score_matrix):


    hit = 0
    hit_count = 0
    miss = 0
    miss_count = 0

    # traverse rows
    for row in range(score_matrix.shape[0]):

        primary_cat = rec_list[row].cat_ID
        primary_title = rec_list[row].image_title
        #print("Cat_ID: {0}; Image: {1}".format(primary_cat, primary_title))

        # traverse columns
        for column in range(score_matrix.shape[1]):
            # dont check the same image.
            if (row != column):
                secondary_cat = rec_list[column].cat_ID

                # Pull the 'hit' out of the score matrix
                if (primary_cat == secondary_cat):
                    hit = hit + score_matrix[row][column]
                    hit_count = hit_count + 1
                else:
                    miss = miss + score_matrix[row][column]
                    miss_count = miss_count + 1

    try:
        print("Hits: {0}; Avg. Hit: {1}".format(hit_count, hit/hit_count))
    except ZeroDivisionError:
        print("Hits: 0; Avg. Miss: 0")

    try:
        print("Misses: {0}; Avg. Miss: {1}".format(miss_count, miss/miss_count))
    except ZeroDivisionError:
        print("Misses: 0; Avg. Miss: 0")
################################################################################
def markov_cluster(mark_array):
    
    # Send square matrix to markov clustering algorithm
    result = mc.run_mcl(mark_array, inflation = 1.5)
    # run MCL with default parameters
    clusters = mc.get_clusters(result)
    #print("results of cluster: ", result)
    print("clusters: ", clusters)
    cluster_array = []
    cluster_array = np.asarray(clusters)
    print("size of cluster: ", cluster_array.shape)
    
    # Test to choose the best inflation point
    # for inflation in [i/10 for i in range (15,26)]:
    #     result = mc.run_mcl(mark_array, inflation = inflation)
    #     clusters = mc.get_clusters(result)
    #     q = mc.modularity(mark_array=result, clusters=clusters)
    #     print("Inflation: ", inflation, "modularity: ", q)
    
    #mc.draw_graph(mark_array, clusters, node_size=50, with_labels=True, edge_color="silver")   
    print("Successful MCL")

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
    
    # sending score matrix with true values to MCL
    markov_cluster(score_matrix_copy)

    return score_matrix

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
    image_path = image_destination.joinpath(str(len(good_points)) +
    "___" +
    re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
    "___" +
    re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title))
    )

    # Finally, write the finished image to the output folder.
    cv2.imwrite(str(image_path), result_vertical_concat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
################################################################################
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the variance
    # of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
################################################################################
def histogram_equalization(image):
    gray =  cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist_image = cv2.equalizeHist(gray)

    print("done with equalization")
    return hist_image
################################################################################  
def edge_sharpening(image):
    kernel = np.array([[-1,-1,-1,],[-1,9,-1],[-1,-1,-1]])
    sharp_image = cv2.filter2D(np.asarray(image), -1, kernel)
    
    return sharp_image
################################################################################
def red(intensity):
    iI = intensity
    i_min = 86
    i_max = 230
    
    o_min = 0
    o_max = 255
    
    #io = ((iI-i_min)/(i_max-i_min))*255
    io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io

def green(intensity):
    iI = intensity
    i_min = 90
    i_max = 225
    
    o_min = 0
    o_max = 255
    
    #io = ((iI-i_min)/(i_max-i_min))*255
    io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io

def blue(intensity):
    iI = intensity
    i_min = 100
    i_max = 210
    
    o_min = 0
    o_max = 255
    
    #io = ((iI-i_min)/(i_max-i_min))*255
    io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io


def filter_images(primary_image,image_source,edited_source):
  
    img_yuv = cv2.cvtColor(primary_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)


    lut_u, lut_v = make_lut_u(), make_lut_v()
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    
    
    min = 125
    max = 130
    i = 530
    #Only checking mid range of each photo for variance from YUV range
    while(i < 550):
        if(u[i][1][1] < 128 or u[i][1][1]>128):

            #Blur score 
            gray = cv2.cvtColor(primary_image, cv2.COLOR_BGR2GRAY)
            threshold = variance_of_laplacian(gray)

            if(threshold < 1500):
                print("not blurry")
            else:
                print("blurry")
        else:
            flag = 1;
            print("night")
            
            
            # img = Image.open(image_source)

            # multi = img.split()
            # red_band = multi[0].point(red)
            # green_band = multi[1].point(green)
            # blue_band = multi[2].point(blue)

            # normal_img = Image.merge("RGB",(red_band, green_band, blue_band))
            # #normal_img.show()
            # save_image = normal_img
            # save_image.save(image_source, "JPEG")
            #normal_img.save(image_source, "JPEG")

            save_image = np.copy(primary_image)
            image_path = image_source
            image_path = image_path.split("\\")

            num = image_path.index('images')
            # Writing original image to folder and editing copy
            edited_source = edited_source + "\\" + image_path[num+1]
            cv2.imwrite(edited_source,primary_image)
            
            sharp_image = edge_sharpening(save_image)
            hist_image = histogram_equalization(sharp_image)
            cv2.imwrite(image_source,hist_image)
            
            #sharp_image = edge_sharpening(normal_img)
            
            # cv2.imshow("final image", np.asarray(hist_image))
            # cv2.waitKey(0)
            break
        i += 1
    
################################################################################
def call_cluster(arr):
    
    # Normalizing the descriptor values before being sent to MCL
    max_num = arr.max()
    arr = arr/max_num
    markov_cluster(arr)
    

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
    


    final_array = np.asarray(descriptors_MCL)

    x,y = final_array.shape
    size = x-y
 
    one_array = np.ones((x,size))
    
    resize_amt = math.floor(math.sqrt(x*y))
    new_array = np.resize(final_array,(resize_amt,resize_amt))
    call_cluster(new_array)
    
    # Markov clustering with one filled array to square matrix
    mark_array = np.hstack((final_array, one_array))
    #markov_cluster(mark_array)
    
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
def add_cat_ID(rec_list, cluster_path):

    # create the list
    import pandas as pd
    csv_file = pd.read_csv(cluster_path)
    image_names = list(csv_file['Image Name'])
    cat_ID_list = list(csv_file['Cat ID'])

    for count in range(len(rec_list)):
        image = os.path.basename(rec_list[count].image_title)
        try:
            image_index = image_names.index(image)
        except ValueError:
            print('\tSomething is wrong with cluster_table file. Image name is not present.')

        cat_ID = cat_ID_list[image_index]
        rec_list[count].add_cat_ID(cat_ID)

    return rec_list

################################################################################
def crop(event, x, y, flags, param):

    global ref_points, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:

        ref_points.append((x, y))
        cropping = False

        cv2.rectangle(param, ref_points[0], ref_points[1], (0, 255, 0), 2)
        cv2.imshow("image", param)

################################################################################
def manual_roi(rec_list, image_source):

    cropping = False
    count = 0

    #todo: do a check and make sure temp_templates folder isn't already created
    temp_templates = image_source.parents[1] / "temp_templates/"
    
    if (not os.path.exists(temp_templates)):
        os.mkdir(temp_templates)

    for i in glob.iglob(str(image_source)):
        print(i)
        # read in image and change size so it doesn't expand to whole screen; make a copy
        image = cv2.imread(i)
        
        image = cv2.resize(image, (960, 540))
        rec_list[count].add_image(i, image)
        image_clone = image.copy()

        # create an empy numpy array for the mask. Will be all black to start
        mask = np.zeros(image.shape, dtype = np.bool)

        # set up event callback for mouse click
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", crop, image)

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                image = image_clone.copy()

            if key == ord("c"):
                break

        if len(ref_points) == 2:

            # Using the coordinates from the moust click. Thurn an area of the mask to white.
            mask[ref_points[0][1]:ref_points[1][1], ref_points[0][0]:ref_points[1][0]] = True
            image = image_clone * (mask.astype(image_clone.dtype))
            locations = np.where(image != 0)
            image[locations[0], locations[1]] = (255, 255, 255)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            template_name = Path(i).with_suffix('.BMP')
            print(template_name.name)
            template_path = temp_templates / template_name.name
            print(template_path)
            cv2.imwrite(str(template_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            rec_list[count].add_template(str(template_path), image)

        cv2.destroyAllWindows()
        count = count + 1

    return rec_list
################################################################################

def mrcnn_templates(rec_list, image_source, snowleop_dir, weights_path):
    'Used for generating templates with the Mask R-CNN and adding'
    'to the recognition class.'
    # MASK R-CNN
    #------------------------------------------------------------------------------
    # This is the Mask R-CNN Function. It takes in the rec_list, the list of recognition objects,
    # the image source or the folder where the images you want to add masks are located, the snow leopard 
    # The snowleop_dir which is the directory of the snow leopard photos that have been trained on, and the weight path
    # which is where the weight (the bottle....h5 file) is located. The snowleop directory is needed because the weight
    # is trained on that dataset. There are more comments underneath here. Up until the for loop is being called, the 
    # Mask RCNN is still being configured. If there are more than 1 snow leopard templated then after the first one,
    # the rest will turn into blank 0's. Make sure you Edit all the paths to reflect your directory or else it will not work.
    #-------------------------------------------------------------------------------
    config = snow_leopard.CustomConfig()
    ## TODO: change this path or get it into easy_run.py
    ##snowleop_dir = os.path.join(ROOT_DIR, "C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/samples/snow_leopard/dataset")
    class InferenceConfig(config.__class__):
    # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    # Device to load the neural network on. Useful if you're training a model on the same machine,
    # in which case use CPU and leave the GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = "inference"
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    # Load validation dataset
    dataset = snow_leopard.CustomDataset()
    dataset.load_custom(snowleop_dir, "val")

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
   
    # Set path to balloon weights file

    # Optional: Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    # weights_path = "/path/to/mask_rcnn_balloon.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(str(weights_path), by_name=True)

    temp_templates = image_source.parents[1] / "mrcnn_templates/"
    if (not os.path.exists(temp_templates)):
        os.mkdir(temp_templates)

    count = 0

    # add in template
    for t in glob.iglob(str(image_source)):
        
        ## TODO: edit this path
        IMAGE_DIR = str(image_source) + '/*'
        # Load a random image from the images folder
        file_names = next(os.walk(IMAGE_DIR))[2]
        maskimage = skimage.io.imread(rec_list[count].image_title)
        # Run detection
        results = model.detect([maskimage], verbose=1)

        # Visualize results
        r = results[0]
        ax = get_ax(1)
        bbb=visualize.display_instances(maskimage, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        display_images(np.transpose(r['masks'], [2, 0, 1]), cmap="binary")
       
        ## TODO: Turn this into a user option of "Is there a cat in this photo?"
        ##       or "Is there more than one cat in this photo?" then either let 
        ##       the Mask R-CNN keep both masks or have the user draw a manual
        ##       template for that image.
        if (np.size(r['masks']) == 0):
            print('\n\tNo cat detected by Mask R-CNN in image.')
            print('\tMaking empty template for this image:',rec_list[count].image_title,'\n\n')
            r_mask = np.zeros(np.shape(rec_list[count].image))
        elif (np.shape(r['masks'])[2] > 1):
            print('\tShape r[''masks'']:', np.shape(r['masks']))
            print('\n\tMore than one cat detected by Mask R-CNN in image:',np.shape(r['masks'])[2],"cats.")
            print('\tMaking empty template for this image:',rec_list[count].image_title,'\n\n')
            r_mask = np.zeros(np.shape(rec_list[count].image))
            #print('\tUsing 1st template generated for this image:',rec_list[count].image_title,'\n\n')
            #r_masks = np.split(r['masks'],np.shape(r['masks'])[2])
            #r_mask = np.reshape(r_masks[1], np.shape(r['masks'])[:2])
        else:
            r_mask = np.reshape(r['masks'], np.shape(r['masks'])[:2])

        r_mask = r_mask * 255

        # get template name and write BMP from r_mask
        template = cv2.imread(t)
        template_name = Path(t).with_suffix('.BMP')
        template_path = temp_templates / template_name.name
        cv2.imwrite(str(template_path), r_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # add template to corresponding rec_list object
        rec_list[count].add_template(str(template_path), r_mask)

        count = count + 1

    return rec_list


################################################################################
def add_templates(rec_list, template_source):
    'Used for adding the premade templates to the recognition class if'
    'the user has them.'
    
    #Ensuring templates and images being matched are the same
    count = 0 
    num = 0 
    num1 = 0 
    countNum = 0
    status = True
    status1 = False
    
    # add in template
    for t in glob.iglob(str(template_source)):
        status1 = False
        template = cv2.imread(t)

        countNum = 0
        while(status1 == False):
            temp = t.split(".")
            temp1 = rec_list[countNum].image_title
            temp1 = temp1.split(".")
            
            temp = temp[0]
            temp1 = temp1[0]
            
            temp = temp.split("/")
            temp1 = temp1.split("/")
            
            num = temp.index("templates")
            num1 = temp1.index("images")
            
            if(temp[num+1] == temp1[num1+1]):
                rec_list[countNum].add_template(t, template)
                count = count + 1
                status1 = True
            countNum = countNum + 1
            count = 0
            
        status = True;
        print("template: ",rec_list[countNum-1].template_title, count)
        print("image: ", rec_list[countNum-1].image_title, count)
        

        # This section is for the presentation only, remove later
        # temp1 = cv2.resize(rec_list[count].image, (960, 540))
        # temp2 = cv2.resize(template, (960, 540))


        # horiz = np.hstack((temp1, temp2))
        # cv2.imshow("mathced image to tempalte", horiz)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # end 

        count = count + 1

    return rec_list


################################################################################
def getTitleChars(title):
    'Used to pull the characteristics out of the image title name'
    title_chars = title.split("__")
    station = title_chars[1]
    camera = title_chars[2]
    date = title_chars[3]
    # dont want the last 7 characters
    time = title_chars[4][:-7]

    return station, camera, date, time
################################################################################
def init_Recognition(image_source, template_source):
    'Used to initalize a recongition object for each template/image pair'
    # # TODO: create a function that verifies the image and template names match

    rec_list = []
    count = 0

    # add images and templates in a parallel for-loop
    for i in glob.iglob(str(image_source)):

        # add new Recognition object to list
        rec_list.append(Recognition())

        # add image title and image to object
        image = cv2.imread(i)
        
        rec_list[count].add_image(i, image)
        #rec_list[count].add_template(rec_list, template_source)

        filter_images(image,i,str(paths['edited_photos']))

        # get title characteristics
        station, camera, date, time = getTitleChars(i)
        rec_list[count].add_title_chars(station, camera, date, time)

        # increment count
        count = count + 1

    # return the list of recognition objects
    return rec_list

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config.ini")
    args = config['default']

    paths = dict()
    paths['images'] = args['image_source']
    paths['templates'] = args['template_source']
    paths['config'] = args['config_source']
    paths['cluster'] = None
    paths['destination'] = args['destination']
    paths['validation_dataset'] = args['validation_dataset']
    paths['weight_source'] = args['weight_source']
    paths['score_matrixCSV'] = args['score_matrixCSV']
    paths['edited_photos'] = args['edited_photos']
    n_threads = int(args['num_threads'])
    write_threshold = int(args['write_threshold'])

    # TODO: change this to fit new command line argument scheme
    # Use the config.json file to import variable parameters
    with open(paths['config']) as config_file:
        parameters = json.load(config_file)

    # initialize the array of Recognition objects for the images
    rec_list = init_Recognition(paths['images'], paths['templates'])

    if (int(parameters['config']['templating']) == 1):
        print('\n\tUsing premade templates...\n')
        rec_list = add_templates(rec_list, paths['templates'])
    elif (int(parameters['config']['templating']) == 2):
        print("\n\tstarting Mask R-CNN templating process...\n")
        start_one = time.time()
        rec_list = mrcnn_templates(rec_list, paths['images'], paths['validation_dataset'],paths['weight_source'])
        end_one = time.time()
        print("\tTime took to run Mask R-CNN: " + str((end_one - start_one)))
    else:
        print('\n\tUsing the manual templating function...\n')
        rec_list = manual_roi(rec_list, paths['images'])

    # Get cat_ID information from cluster table
    if (paths['cluster'] != None):
        print("\n\tLoading information from cluster table...\n")
        # add in the cat ID data if available
        rec_list = add_cat_ID(rec_list, paths['cluster'])
        
    lock = threading.Lock()
    
    
    # START
    print("\tstarting matching process...\n")
    start = time.time()
    for rec in rec_list:
        rec.calculate_kp()
    score_matrix = match_multi(rec_list, paths['destination'], n_threads, write_threshold, parameters)
    print("Type kp:", type(rec_list[0].key_points))
    print("Type desc:", type(rec_list[0].kp_desc))


    # Normalize scores in matrix
    score_matrix = normailze_matrix(score_matrix)
    
    # write the score matrix to a .csv file
    print("\n\tWriting score_matrix to 'score_matrix.csv' to the destination folder...\n")
    np.savetxt(paths['destination']+'/score_matrix.csv', score_matrix, delimiter = ",")


    # check matrix for average hit/miss scores
    if (paths['cluster'] != None):
        print("Checking matrix...")
        check_matrix(rec_list, score_matrix)
    
    kmeans_score_matrix(paths['score_matrixCSV'])
    
    end = time.time()
    
    print("\tTime took to run: " + str((end - start)))

    print('\n\tDone.\n')
    
############################### END  MAIN ######################################
