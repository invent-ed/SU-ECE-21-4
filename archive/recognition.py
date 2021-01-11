import configparser

import os, sys

import threading
import time, datetime
import json
import numpy as np

import match
import imageProcessing
import templating
global mark_array
########################### END IMPORTS ########################################



################################################################################
################## 			MAIN 			 				 ###################
################################################################################
if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config.ini")
    args = config['edward']

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
    rec_list = imageProcessing.init_Recognition(paths['images'], paths['templates'],paths)

    if (int(parameters['config']['templating']) == 1):
        print('\n\tUsing premade templates...\n')
        rec_list = templating.add_templates(rec_list, paths['templates'])
    elif (int(parameters['config']['templating']) == 2):
        print("\n\tstarting Mask R-CNN templating process...\n")
        start_one = time.time()
        rec_list = templating.mrcnn_templates(rec_list, paths['images'], paths['validation_dataset'],paths['weight_source'])
        end_one = time.time()
        print("\tTime took to run Mask R-CNN: " + str((end_one - start_one)))
    else:
        print('\n\tUsing the manual templating function...\n')
        rec_list = templating.manual_roi(rec_list, paths['images'])

    # Get cat_ID information from cluster table
    if (paths['cluster'] != None):
        print("\n\tLoading information from cluster table...\n")
        # add in the cat ID data if available
        rec_list = imageProcessing.add_cat_ID(rec_list, paths['cluster'])
        
    lock = threading.Lock()
    
    
    # START
    print("\tstarting matching process...\n")
    start = time.time()
    for rec in rec_list:
        rec.calculate_kp()
    score_matrix = match.match_multi(rec_list, paths['destination'], parameters)
    print("Type kp:", type(rec_list[0].key_points))
    print("Type desc:", type(rec_list[0].kp_desc))


    # Normalize scores in matrix
    score_matrix = match.normailze_matrix(score_matrix)
    
    # write the score matrix to a .csv file
    print("\n\tWriting score_matrix to 'score_matrix.csv' to the destination folder...\n")
    np.savetxt(paths['destination']+'/score_matrix.csv', score_matrix, delimiter = ",")
        
    end = time.time()
    
    print("\tTime took to run: " + str((end - start)))

    print('\n\tDone.\n')
    
############################### END  MAIN ######################################
