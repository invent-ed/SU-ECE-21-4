import os
import logging
from AbstractBaseClass.GroupStage import GroupStage
from ConcreteClass.Group import Group


class OrphanGroupStage(GroupStage):

    def __init__(self, config, Matcher):
        self.config = config
        self.Matcher = Matcher

    def process(self, groups_list):
        return groups_list
