class Group():
	def __init__(self, groupNames):
		self.filenames = groupNames
		self.representative_indices = []
		self.grouped_list_indices = []

	#return keypoints of the actual representatives