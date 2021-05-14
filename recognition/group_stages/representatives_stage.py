import logging


# This stage also returns the list of groups.
# During the match group functions, the find representatives function is called on each group.
# The keypoints of the representative images are loaded into the program for the matching process.
# Matching is determined by the matcher object, so can be changed with new child class.
def group_by_representatives(matcher, groups_list):
    numShouldMatch = 0
    for i, prim in enumerate(groups_list): 
        for j, sec in enumerate(groups_list):
            if j > i:
                if Matcher.matchCheck(prim.filenames[0], sec.filenames[0]):
                    numShouldMatch = numShouldMatch + 1
    print("Number of groups:",len(groups_list), ". Num should match:",numShouldMatch)
    
    rep_kps_list = load_keypoints_for_each_representative(groups_list)
    matched_pairs_list = find_groups_with_matched_representatives(rep_kps_list, matcher)
    merged_sets_list = sets_of_merged_groups(matched_pairs_list)
    complete_sets_list = include_orphans(merged_sets_list, len(groups_list))
    new_groups = generate_new_groups_from_sets(groups_list, complete_sets_list)
    return new_groups


# Input:  list of Group objects
# Output: 2D list (matrix), rows for each group, columns for each representative in that group
#         Each entry in the matrix is the keypoints (kps) object for each representative (rep) for each group
#         e.g.  [ [kps of rep image 1, kps of rep image 2, ....],
#                 [kps of rep image 1],
#                           .
#                           .
#                 [kps of rep image 1, kps of rep image 2, ....] ]
#         Note: not all groups will have the same number of representatives (ex: metadata group of size 1)
def load_keypoints_for_each_representative(groups_list):
    logging.info("Beginning to load representative keypoints")
    rep_kps_list = []
    for group in groups_list:
        group.find_representatives()
        rep_kps = group.get_rep_keypoints()
        rep_kps_list.append(rep_kps)
    logging.info("Finished loading representative keypoints")
    return rep_kps_list


def find_groups_with_matched_representatives(rep_kps_list, matcher):
    matched_groups = []
    # Loops through every metadata group. (This loop creates the primary metadata group being compared)
    # Loops through all of the keypoint objects, one for each representative, of the primary metadata group.
    # Loops to a different metadata group. (Secondary metadata group. Different than the primary group.)
    # Compares all of the keypoint objects of the secondary metadata group to current keypoint object
    # of the primary keypoint group.
    ijMatched = False
    for i, primary_group_reps in enumerate(rep_kps_list):
        for primaryKpsObj in primary_group_reps:
            for j, secondary_group_reps in enumerate(rep_kps_list):
                if j > i and not ijMatched:
                    for secondaryKpsObj in secondary_group_reps:
                        logging.info("Checking two groups ")
                        if matcher.match(primaryKpsObj, secondaryKpsObj):
                            # Records a match as a tuple of group indices
                            matched_groups.append((i, j))
                            logging.info("Matched: two groups")
                            ijMatched = True
                        else:
                            # Records the group as being matched to itself.
                            # This is so that every group is accounted for when creating the matched groups object.
                            matched_groups.append((i, i))
        ijMatched = False
    return matched_groups


# Input:  List of tuples containing the indices of the pairs of Groups to be merged
#         e.g. sets_of_merged_groups([(0,1), (2,3), (4,1), (5,6), (8,6), (8,11)])
# Output: List of sets containing the indices of the merged Groups
#         e.g. [{2, 3}, {0, 1, 4}, {5, 6, 8, 11}]
def sets_of_merged_groups(matched_pairs_list):
    indices = map(set, matched_pairs_list)
    unions = []
    for item in indices:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions


# Input:  List of sets of Groups to be merged, and total number of Groups
#         e.g. include_orphans([{2, 3}, {0, 1, 4}, {5, 6, 8, 11}], 12)
# Output: Final list of sets containing the indices of the merged Groups AND orphans
#         e.g. [{2, 3}, {0, 1, 4}, {5, 6, 8, 11}, {9}, {10}, {7}]
def include_orphans(merged_sets_list, total):
    all_groups = set(range(total))
    non_orphans = set.union(*merged_sets_list)
    orphans = all_groups - non_orphans
    orphans_list = [set([orphan]) for orphan in orphans]
    return merged_sets_list + orphans_list

# Input1: old list of Group objects
#         e.g. [group0, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11]
# Input2: list of sets of indices of Group objects to be merged
#         e.g. [{2, 3}, {0, 1, 4}, {5, 6, 8, 11}, {9}, {10}, {7}]
# Output: new list of Group objects
#         e.g. [newGroup0, newGroup1, newGroup2, newGroup3, newGroup4, newGroup5]
#              For example, to make newGroup0, it merges the Groups with the indices listed in the 0th set
#              So group2 would be the primary group that the secondary group group3 merges into
#              After merging all the secondary groups, group2 becomes newGroup0
#              Similarly, newGroup1 is created by merging secondary groups group1 and group4 into primary group group0
def generate_new_groups_from_sets(groups_list, complete_sets_list):
    new_groups = []
    list_of_old_group_indices = [list(s) for s in complete_sets_list]
    for old_group_indices in list_of_old_group_indices:
        primary_group = groups_list[old_group_indices[0]]
        for secondary_group_index in old_group_indices[1:]:
            secondary_group = groups_list[secondary_group_index]
            primary_group.merge_with(secondary_group)
        new_groups.append(primary_group)
    return new_groups

