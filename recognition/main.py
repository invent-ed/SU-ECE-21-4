# Panthera Open Source Software Recognition
# Created by Seattle University ECE 21.4
# Gao, Edward; Pham, Dustin; Sadaoka Neil; Somlak, Calien.
# Previous version of Recognition: https://github.com/caballe4/SU-ECE-20-4

import os
import csv
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.SiftKeypoints import SiftKeypoints
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator
from ConcreteClass.SiftKeypointsMatcher import SiftKeypointsMatcher
from group_stages.metadata_stage import group_by_metadata
from group_stages.representatives_stage import group_by_representatives

    
def write_groups_to_csv(catID_csv, groups_list):
    with open (catID_csv, 'w') as file:
        writer = csv.writer(file)
        for i, group in enumerate(groups_list):
            for file in group.filenames:
                writer.writerow([file + ".JPG", "Cat_Group_" + str(i)])
        
 
if __name__ == "__main__":

    # set up
    config = JsonConfig("data/config.json")
    config.setup_logger()
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config, maskGenerator)
    matcher = SiftKeypointsMatcher(config)

    # generate keypoints for each image if they do not exist
    for image_path in config.get_image_list():
        kps_path = SiftKeypoints.generate_keypoints_path(config, image_path)
        if not os.path.isfile(kps_path):
            imageObj = SnowLeopardImage(image_path)
            keypointsGenerator.generate_and_save_keypoints(imageObj, kps_path)

    # run through each group stage
    groups_list = []
    groups_list = group_by_metadata(config, groups_list)
    groups_list = group_by_representatives(matcher, groups_list)
    
    write_groups_to_csv(config.get("results.catID_csv"), groups_list)

    

 
        