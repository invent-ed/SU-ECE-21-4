import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group

def match_groups(config, Matcher, groups_list):
	logging.info("Beginning to collect representative keypoints")
	rep_kps_list = []
	for group in groups_list:
		rep_kps = group.get_rep_keypoints()
		rep_kps_list.append(rep_kps)
	logging.info("Finished collecting representative keypoints")
		
	matched_groups = []
	logging.info("Starting Matching Process")
	for i, primary_group_reps in enumerate(rep_kps_list):
		for primaryKpsObj in primary_group_reps:
			for j, secondary_group_reps in enumerate(rep_kps_list):
				if j > i:
					for secondaryKpsObj in secondary_group_reps:
						logging.info("Checking two groups ")
						if (Matcher.match(primaryKpsObj, secondaryKpsObj)):
							matched_groups.append((i,j))
							logging.info("Matched: two groups")
						else: 
							matched_groups.append((i,i))
	print(matched_groups)
	logging.info("Finished matching process")
	matched_groups = sets_of_merged_groups(matched_groups)
	matched_groups_list = []
	print("Matched groups:",matched_groups)
	for i in matched_groups:
		i = list(i)
		for j in i[1:]:
			groups_list[i[0]].merge_groups(groups_list[j])
		matched_groups_list.append(groups_list[i[0]])

	print("Number of groups:",len(matched_groups_list))
	return matched_groups_list
	
def sets_of_merged_groups(matched_groups):
	indicies = map(set, matched_groups)
	unions = []
	for item in indicies:
		temp = []
		for s in unions:
			if not s.isdisjoint(item):
				item = s.union(item)
			else:
				temp.append(s)
		temp.append(item)
		unions = temp
	return unions