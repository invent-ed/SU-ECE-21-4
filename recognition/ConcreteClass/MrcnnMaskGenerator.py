import os
import logging
import numpy as np
import tensorflow as tf
import mrcnn.model as modellib
from mrcnn.config import Config
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
        self.load_mrcnn_model()

    def initialize_mrcnn_config(self):
        logging.info("Initializing MRCNN config")
        self.mrcnn_config = InferenceConfig()
        self.mrcnn_config.display()

    def load_mrcnn_model(self):
        logging.info("Loading MRCNN model")
        model_dir = self.config.get("Mask.mrcnn.model_directory")
        weights_path = self.config.get("Mask.mrcnn.weights_path")
        with tf.device(self.config.get("Mask.mrcnn.device")):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=self.mrcnn_config)
        self.model.load_weights(weights_path, by_name=True)

    def generate_mask_if_not_exist(self, imageObj):
        logging.info("Generating a mask if it does not already exist")
        mask_path = MaskImage.generate_mask_path(self.config, imageObj.filename)
        if not os.path.isfile(mask_path):
            self.generate_and_save_mask(imageObj, mask_path)
        return mask_path

    def generate_and_save_mask(self, imageObj, mask_path=None):
        logging.info("Generating mask")
        if mask_path is None:
            mask_path = MaskImage.generate_mask_path(self.config, imageObj.filename)
        masks = np.array(self.model.detect([imageObj.image], verbose=1)[0]['masks'])
        mask = self.collapse_multiple_masks_to_one(masks)
        MaskImage.save_mask_to_file(mask_path, mask)

    def collapse_multiple_masks_to_one(self, masks):
        logging.info("Collapsing color channels")
        size = np.shape(masks)
        mask = np.dot(masks, [[1]] * size[2])
        mask = np.reshape(mask, size[:2])
        return np.uint8(mask * 255)


class InferenceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "snow_leopard"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Inference
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    PRE_NMS_LIMIT = 6000
