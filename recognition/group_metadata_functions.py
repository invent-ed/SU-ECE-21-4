import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group

def group_by_metadata(config, groups_list):
    logging.info("Grouping by metadata")
    list_of_groups = []
    grouped_images = []
    prev_hour = 0
    prev_minute = 0
    prev_sec = 0
    prev_station = None
    prev_camera = None
    prev_date = None
    for image_path in config.get_image_list():
        station, camera, date, time = extract_camera_trap_info(image_path)
        hour, minute, sec = extract_time_info(time)
        # Exceptions if hour is near the edge
        if hour == 0 and prev_hour == 23:
            hour += 24
        # if previous image is 1 hour behind
        if (hour - prev_hour) == 1:
            new_minute = minute + 60
            time_difference = new_minute - prev_minute
        # end of exception its in the same hour
        elif hour == prev_hour:
            time_difference = minute - prev_minute
        else:
            time_difference = 10

        # Group by station, camera, date, and time (within 5 min)
        # Or first item in new group.
        if ((station == prev_station) and (camera == prev_camera) and (date == prev_date) and (
                time_difference < 6)) or (len(grouped_images) == 0):
            grouped_images.append(filename_without_ext(image_path))
            prev_station, prev_camera, prev_date, prev_hour, prev_minute, prev_sec = station, camera, date, hour, minute, sec
        # If not, add list to Group, append Group to list, and create new list
        else:
            newGroup = Group(config, grouped_images)
            list_of_groups.append(newGroup)
            # Create new list
            grouped_images = []
            prev_station, prev_camera, prev_date, prev_hour, prev_minute, prev_sec = station, camera, date, hour, minute, sec

    newGroup = Group(config, grouped_images)
    list_of_groups.append(newGroup)
    logging.info("All metadata groups have been made")
    return list_of_groups

def filename_without_ext(image_path):
    base = os.path.basename(image_path)
    return os.path.splitext(base)[0]

def extract_camera_trap_info(image_path):
    logging.info("Extracting camera trap info")
    filename = filename_without_ext(image_path)
    camera_info = filename.split("__")
    station = camera_info[1]
    camera = camera_info[2]
    date = camera_info[3]
    time = camera_info[4][:-3]
    return station, camera, date, time

def extract_time_info(time):
    time_chars = time.split("-")
    hour = time_chars[0]  # hour 0 - 23
    minute = time_chars[1]  # minutes 0-59
    sec = time_chars[2]  # seconds 0-59
    return int(hour), int(minute), int(sec)