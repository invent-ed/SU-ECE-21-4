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
