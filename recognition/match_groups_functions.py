import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group

def match_groups(config, Matcher, groups_list):
	logging.info("Beginning to collect representative keypoints")

	#count all of the groups that should be matched:
	numShouldMatch = 0
	for i, prim in enumerate(groups_list): 
		for j, sec in enumerate(groups_list):
			if j > i:
				if Matcher.matchCheck(prim.filenames[0], sec.filenames[0]):
					numShouldMatch = numShouldMatch + 1
	print("Number of groups:",len(groups_list), ". Num should match:",numShouldMatch)

	#Finds representatives based on images with the most keypoints. 
	#Creates list rep_kps_list[].
	#This list is 2D, and follows this format:
	#metadata group 0 at the 0 index, metadata group 1 at the 1st index in the list...
	#Each index holds a list of they keypoint objects of the representative images for that metadata group.	
	#    [[kps of rep image 1, kps of rep image 2, ....],
	#	  [kps of rep image 1],
	#							.
	#							.
	#	  [kps of rep image 1, kps of rep image 2, ....]]
	#Note: not all groups will have the same number of representatives (ex: metadata group of size 1)
	rep_kps_list = []
	for group in groups_list:
		group.find_representatives()
		rep_kps = group.get_rep_keypoints()
		rep_kps_list.append(rep_kps)
	logging.info("Finished collecting representative keypoints")
		
	matched_groups = []
	logging.info("Starting Matching Process")

	#Loops through every metadata group. (This loop creates the primary metadata group being compared)
	#Loops through all of the keypoint objects, one for each representative, of the primary metadata group.
	#Loops to a different metadata group. (Secondary metadata group. Different than the primary group.)
	#Compares all of the keypoint objects of the secondary metadata group to current keypoint object
	#of the primary keypoint group. 

	ijMatched = False
	for i, primary_group_reps in enumerate(rep_kps_list):
		for primaryKpsObj in primary_group_reps:
			for j, secondary_group_reps in enumerate(rep_kps_list):
				if j > i and not ijMatched:
					for secondaryKpsObj in secondary_group_reps:
						logging.info("Checking two groups ")
						if (Matcher.match(primaryKpsObj, secondaryKpsObj)):
							#Records a match as a touple of group indicies. 
							matched_groups.append((i,j))
							logging.info("Matched: two groups")
							ijMatched = True
						else: 
							#Records the group as being matched to itself. 
							#This is so that every group is accounted for when creating the matched groups object. 
							matched_groups.append((i,i))
		ijMatched = False

	print(matched_groups)
	logging.info("Finished matching process")
	matched_groups = sets_of_merged_groups(matched_groups)
	print(matched_groups)

	#Go through the list of all the matches. (If no match the single group index will just be in the list.)
	#Merge all of the groups that have been matched together. 
	#Append the single metadata group, or the group that has been merged to. 
	matched_groups_list = []
	for i in matched_groups:
		i = list(i)
		for j in i[1:]:
			groups_list[i[0]].merge_groups(groups_list[j])
		matched_groups_list.append(groups_list[i[0]])

	return matched_groups_list
	
#Returns the list where any touples are combined if they have the same number in any position. 
#Ex: sets_of_merged_groups(  [(0,0) , (0,0), (1,3), (3,4), (2,2), (1,1), (4,4)] )
#    returns: [{0}, {1,3,4}, {2}] (Exactly what we want, only three groups at the end in this case.)
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