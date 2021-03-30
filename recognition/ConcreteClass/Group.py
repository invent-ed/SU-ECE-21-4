import pickle
from ConcreteClass.MaskImage import MaskImage
from ConcreteClass.SiftKeypoints import SiftKeypoints
import numpy as np


class Group:

    def __init__(self, config, filenames):
        self.config = config
        self.filenames = filenames
        self.representative_indices = []
        self.grouped_list_indices = []

    def get_rep_masks(self):
        rep_mask_objs = []
        for i in self.representative_indices:
            mask_path = MaskImage.generate_mask_path(self.filenames[i])
            rep_mask_objs.append(MaskImage(mask_path))
        return rep_mask_objs

    def get_rep_keypoints(self):
        rep_keypoint_objs = []
        for i in self.representative_indices:
            kps_path = SiftKeypoints.generate_keypoints_path(self.config, self.filenames[i])
            rep_keypoint_objs.append(SiftKeypoints(kps_path))
        return rep_keypoint_objs

    def find_number_of_keypoints_all_images(self):
        num_kps_all_images = []
        for filename in self.filenames:
            pickle_file = open(SiftKeypoints.generate_keypoints_path(self.config, filename), "rb")
            kps_and_descs_list = pickle.load(pickle_file)
            pickle_file.close()
            num_kps_all_images.append(len(kps_and_descs_list))
        return num_kps_all_images

    def find_representatives(self):
        #initially keypoints  - mask - blurriness...
        num_kps_all_images = self.find_number_of_keypoints_all_images()
        num_kps_all_images = np.array(num_kps_all_images)
        print(num_kps_all_images)
        order = np.argsort(num_kps_all_images)
        #for i in range(1,3):
        self.representative_indices.append(order[-1])
        print(self.representative_indices, num_kps_all_images[order[-1:]])

