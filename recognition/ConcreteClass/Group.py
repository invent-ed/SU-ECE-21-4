class Group():
	def __init__(self, config, groupNames):
		self.config = config
		self.filenames = groupNames
		self.representative_indices = []
		self.grouped_list_indices = []

	#return keypoints of the actual representatives
    def load_rep_keypoints(self):
    	representative_kps = []
    	representative_desc = []
    	for i in representative_indices: 
    		keypoints, descriptors = read_kps_desc_from_file(generate_keypoints_path(filenames[i]))
	        representative_kps.append(keypoints)
	        representative_desc.append(descriptors)
	    return representative_kps, representative_desc

    def load_all_keypoints(self):
    	all_kps = []
    	all_desc = []
    	for filename in filenames: 
    		keypoints, descriptors = read_kps_desc_from_file(generate_keypoints_path(filename))
	        all_kps.append(keypoints)
	        all_desc.append(descriptors)
	    return all_kps, all_desc

	def read_kps_desc_from_file(self, kps_path):
		keypoints = []
		descriptors = []
		pickle_file = open(kps_path, "rb")
        kps_and_descs_list = pickle.load(pickle_file)
        pickle_file.close()
        for kp_and_desc in kps_and_descs_list:
            [pt, size, angle, response, octave, class_id, desc] = kp_and_desc
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle, _response=response, _octave=octave, _class_id=class_id)
            keypoints.append(kp)
            descriptors.append(desc)
            del kp_and_desc
        descriptors = np.asarray(descriptors)
	    return keypoints, descriptors 

	def generate_keypoints_path(filename):
		logging.info("Generating keypoints path")
		kp_dir = config.get("Keypoints.directory")
		kp_ext = config.get("Keypoints.file_extension")
		return os.path.abspath(kp_dir).replace("\\", "/") + "/" + filename + kp_ext
	
	#return masks of the actual representatives