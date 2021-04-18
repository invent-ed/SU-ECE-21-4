from abc import ABCMeta, abstractmethod


class GroupStage(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def process(self, groups_list):
        pass

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
