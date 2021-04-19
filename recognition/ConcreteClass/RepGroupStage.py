import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group


class RepGroupStage(GroupStage):

    def __init__(self, config, Matcher):
        self.config = config
        self.Matcher = Matcher

    def process(self, groups_list):
        rep_kps_list = []
        for group in groups_list:
            rep_kps = group.get_rep_keypoints()
            #for i in rep_kps:
            rep_kps_list.append(rep_kps)

        for i, primary_group_reps in enumerate(rep_kps_list):
            for primaryKpsObj in primary_group_reps:
                for j, secondary_group_reps in enumerate(rep_kps_list):
                    if j > i:
                        for secondaryKpsObj in secondary_group_reps:
                            num_strong_matches = self.Matcher.match(primaryKpsObj, secondaryKpsObj)
                            print("Number of strong matches: ", num_strong_matches)
                            logging.info("Number of strong matches: " + str(num_strong_matches))
        #sets_of_merged_groups()

                # Then do some function that will group if conditions are meet

    def sets_of_merged_groups(self, matched_groups):
        groups = map(set, matched_groups)
        unions = []
        for item in groups:
            temp = []
            for s in unions:
                if not s.isdisjoint(item):
                    item = s.union(item)
                else:
                    temp.append(s)
            temp.append(item)
            unions = temp
        return unions