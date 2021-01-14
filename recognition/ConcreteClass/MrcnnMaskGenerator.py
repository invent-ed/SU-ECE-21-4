import os
import numpy as np
import tensorflow as tf
import mask_rcnn.mrcnn.model as modellib
import logging
from mask_rcnn.samples.snow_leopard import snow_leopard
from AbstractBaseClass.MaskGenerator import MaskGenerator
from ConcreteClass.MaskImage import MaskImage


class MrcnnMaskGenerator(MaskGenerator):

    def __init__(self, config):
        self.config = config
        self.mrcnn_config = None
        self.dataset = None
        self.model = None
        self.create_masks_dir_if_not_exist()
        self.initialize()

    def create_masks_dir_if_not_exist(self):
        masks_dir = self.config.get("Mask.directory")
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

    def initialize(self):
        logging.info("Initializing Mask Generator")
        self.initialize_mrcnn_config()
        self.load_snow_leopards_dataset()
        self.load_mrcnn_model()
        

    def initialize_mrcnn_config(self):
        logging.info("Initializing Mask config")
        self.mrcnn_config = snow_leopard.CustomConfig()

        class InferenceConfig(self.mrcnn_config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            PRE_NMS_LIMIT = 6000

        self.mrcnn_config = InferenceConfig()
        self.mrcnn_config.display()

    def load_snow_leopards_dataset(self):
        logging.info("Loading snow leaopards dataset")
        self.dataset = snow_leopard.CustomDataset()
        self.dataset.load_custom(self.config.get("Mask.mrcnn.validation_set"), "val")
        self.dataset.prepare()

    def load_mrcnn_model(self):
        logging.info("Loading mask model")
        model_dir = self.config.get("Mask.mrcnn.model_directory")
        weights_path = self.config.get("Mask.mrcnn.weights_path")
        with tf.device(self.config.get("Mask.mrcnn.device")):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=self.mrcnn_config)
        self.model.load_weights(weights_path, by_name=True)

    def generate_mask_if_not_exist(self, imageObj):
        logging.info("Generating a mask if it does not already exist")
        mask_path = self.generate_mask_path(imageObj.filename)
        if not os.path.isfile(mask_path):
            mask = self.generate_mask(imageObj)
            MaskImage.save_mask_to_file(mask_path, mask)
            mask = None
        return MaskImage(mask_path)

    def generate_mask_path(self, filename):
        logging.info("Generating mask path")
        mask_dir = self.config.get("Mask.directory")
        mask_ext = self.config.get("Mask.file_extension")
        return os.path.abspath(mask_dir).replace("\\", "/") + "/" + filename + mask_ext

    def generate_mask(self, imageObj):
        logging.info("Generating mask")
        masks = np.array(self.model.detect([imageObj.image], verbose=1)[0]['masks'])
        mask = MaskImage.collapse_color_channels(masks)
        return mask
