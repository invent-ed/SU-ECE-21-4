# Returns the list where any touples are combined if they have the same number in any position.
# Ex: sets_of_merged_groups(  [(0,0) , (0,0), (1,3), (3,4), (2,2), (1,1), (4,4)] )
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
