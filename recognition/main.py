#Panthera Open Source Software Recognition
#Main function.
#Created by Seattle University ECE 21.4 
#Gao, Edward; Pham, Dustin; Sadaoka Neil; Somlak, Calien. 
#Previous version of Recognition: https://github.com/caballe4/SU-ECE-20-4

import os
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.SiftKeypoints import SiftKeypoints
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator
from ConcreteClass.SiftKeypointsMatcher import SiftKeypointsMatcher


from group_metadata_functions import *
from group_orphans_functions import *
from match_groups_functions import *


if __name__ == "__main__":

    #Sets up config file and logger for debugging. 
    config = JsonConfig("data/config.json")
    config.setup_logger()

    #Sets up the mask and keypoint generator. 
    #Generator is a class used to generate MRCNN masks and SIFT keypoints
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config)

    #Current matcher matches SIFT keypoints. 
    #Child class should be created for other versions of the match function. 
    matcher = SiftKeypointsMatcher(config)

    # generate mask and keypoints for each image
    for image_path in config.get_image_list():
        kps_path = SiftKeypoints.generate_keypoints_path(config, image_path)
        #If there isn't a keypoints file to load the keypoints from. 
        if not os.path.isfile(kps_path):
            imageObj = SnowLeopardImage(image_path)
            keypointsGenerator.generate_and_save_keypoints(imageObj, kps_path)

    #run through each group stage
    groups_list = []
    #This group stage returns a list of all of the group objects.
    #Each group object at this stage contains all of the image titles that are in a metadata group.
    #Doesn't load in anything other than the image titles. This ensures that Recognition doesn't use too much memory.
    groups_list = group_by_metadata(config, groups_list)
    #This stage also returns the list of groups. 
    #During the match group functions, the find representatives function is called on each group.
    #The keypoints of the representative images are loaded into the program for the matching process. 
    #Matching is determined by the matcher object, so can be changed with new child class. 
    groups_list = match_groups(config, matcher, groups_list)
    

