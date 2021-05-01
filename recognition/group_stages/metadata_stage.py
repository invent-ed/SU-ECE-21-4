import os
import time
import logging
from ConcreteClass.Group import Group


# Returns the initial list of Group objects
# Images in each group are sequentially taken within threshold_time of each other
# threshold_time is defined (in minutes) in the config file: grouping.metadata.threshold
def group_by_metadata(config, groups_list):

    logging.info("Grouping by metadata")
    groups = []
    curr_group = []
    prev_location = None
    prev_epoch_time = None

    for image_path in config.get_image_list():

        # extract info from current image
        filename = filename_without_ext(image_path)
        location, datetime = extract_image_metadata(filename)
        epoch_time = convert_to_epoch_time(datetime)

        # compare with previous image
        threshold_time = config.get("grouping.metadata.threshold") * 60  # 60 sec per min
        image_taken_at_same_location = (location == prev_location)
        within_threshold_time_apart = (epoch_time - prev_epoch_time) <= threshold_time

        # decide which group to put current image in
        if image_taken_at_same_location and within_threshold_time_apart:
            # add current image to current group
            curr_group.append(filename)
        else:
            # add current group to list of groups, and start new group
            groups.append(curr_group)
            curr_group = [filename]
        prev_location, prev_epoch_time = location, epoch_time

    # include last group and return list of Group objects
    groups.append(curr_group)
    return [Group(config, g) for g in groups]


def filename_without_ext(image_path):
    base = os.path.basename(image_path)
    return os.path.splitext(base)[0]


def extract_image_metadata(filename):
    ignored, station, camera, date, hms = filename.split("__")
    location = " ".join([station, camera])
    datetime = " ".join([date, hms])[:-3]
    return location, datetime


def convert_to_epoch_time(datetime):
    return int(time.mktime(time.strptime(datetime, "%Y-%m-%d %H-%M-%S")))
