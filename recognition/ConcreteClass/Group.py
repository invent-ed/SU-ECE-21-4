class Group():
	def __init__(self, config, groupNames):
		self.config = config
		self.filenames = groupNames
		self.representative_indices = []
		self.grouped_list_indices = []

	#return keypoints of the actual representatives