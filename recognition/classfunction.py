import cv2

# CLASS DEFINITION
################################################################################
class Recognition:
    'This class holds an image-template pair.'
    'Keeps pairs secure and together thorughout whole process and'
    'cuts down on code bloat. Also holds the image title split into its'
    'base characterisitics and the proper cat ID'
    def __init__(self):
        self.image_title = ""
        self.image = ""
        self.template_title = ""
        self.template = ""
        self.station = ""
        self.camera = ""
        self.date = ""
        self.time = ""
        self.cat_ID = ""
        self.key_points = None
        self.kp_desc = None

    def add_image(self, image_title, image):
        self.image_title = image_title
        self.image = image

    def add_template(self, template_title, template):
        self.template_title = template_title
        self.template = template

    def add_title_chars(self, title):
    	title_chars = title.split("__")
    	self.station = title_chars[1]
    	self.camera = title_chars[2]
    	self.date = title_chars[3]
    	self.time = title_chars[4][:-7]

    def add_cat_ID(self, cat):
        self.cat_ID = cat
        
    def calculate_kp(self):
        mask_1 = cv2.imread(self.template_title, -1) 
        mySift = cv2.xfeatures2d.SIFT_create()
        self.key_points, self.kp_desc = mySift.detectAndCompute(self.image, mask_1)
########################### END CLASS DEFINITION ###############################