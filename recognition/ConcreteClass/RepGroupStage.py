import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group


class RepGroupStage(GroupStage):

    def __init__(self, config, Matcher):
        self.config = config
        self.Matcher = Matcher

    def process(self, groups_list):
        kps_list = []
        for group in groups_list:
            rep_kps = group.get_rep_keypoints()
            for i in rep_kps:
                kps_list.append(i)

        for i, primaryKpsObj in enumerate(kps_list):
            for j, secondaryKpsObj in enumerate(kps_list):
                if j > i:
                    num_strong_matches = self.Matcher.match(primaryKpsObj, secondaryKpsObj)
                    print("Number of strong matches: ", num_strong_matches)
                    logging.info("Number of strong matches: " + str(num_strong_matches))

                # Then do some function that will group if conditions are meet
