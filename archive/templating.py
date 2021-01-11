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

import skimage.io

#mask r-cnn
import random
import tensorflow as tf
import matplotlib


# EDIT PATHS FOR MASK R-CNN LIBRARY FOLDER ################################################################################
# Root directory of the project
ROOT_DIR = os.path.abspath("/app")

#print(ROOT_DIR)
#ROOT_DIR = "Mask_RCNN-master"

MODEL_DIR = os.path.join(ROOT_DIR, "data/logs")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mask_rcnn.mrcnn import utils
from mask_rcnn.mrcnn import visualize
from mask_rcnn.mrcnn.visualize import display_images
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn.model import log

from mask_rcnn.samples.snow_leopard import snow_leopard

################################################################################
###########        TEMPLATE FUNCTIONS 			 ###############################
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
    #snowleop_dir = os.path.join(ROOT_DIR, "/app/recognition/mask_rcnn/samples/snow_leopard/dataset")
    class InferenceConfig(config.__class__):
    # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        PRE_NMS_LIMIT = 6000

    config = InferenceConfig()
    config.display()
    # Device to load the neural network on. Useful if you're training a model on the same machine,
    # in which case use CPU and leave the GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    #TEST_MODE = "inference"
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
    print("Test 1 - Instance right before defining mrcnn model.\n\n")
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    print("Test 2 - Instance right after defining model.\n\n")
    # Set path to balloon weights file

    # Optional: Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    # weights_path = "/path/to/mask_rcnn_balloon.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(str(weights_path), by_name=True)
    print("Test 3 - Instance right after loading weights. \n\n")
    # d = Path(__file__).resolve().parents[1]
    # print(d)
    #print(image_source)
    temp_templates = "/app/data/mrcnn_templates"
    if not os.path.exists(temp_templates):
        os.makedirs(temp_templates)

    count = 0

    # add in template
    for t in glob.iglob(str(image_source)):
        
        ## TODO: edit this path
        #print('t: ',t)
        #print('Image Source: ', image_source)
        IMAGE_DIR = image_source
        # Load a random image from the images folder
        file_names = t
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
        template_path = temp_templates + '/'+template_name.name
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
            
            #need this for windows, mac does not
            temp = temp.replace("\\","/")
            temp1 = temp1.replace("\\","/")
            
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