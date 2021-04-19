import os
from ConcreteClass.JsonConfig import JsonConfig
from ConcreteClass.SnowLeopardImage import SnowLeopardImage
from ConcreteClass.SiftKeypoints import SiftKeypoints
from ConcreteClass.MrcnnMaskGenerator import MrcnnMaskGenerator
from ConcreteClass.SiftKeypointsGenerator import SiftKeypointsGenerator
from ConcreteClass.SiftKeypointsMatcher import SiftKeypointsMatcher
from ConcreteClass.MetadataGroupStage import MetadataGroupStage
from ConcreteClass.RepGroupStage import RepGroupStage
from ConcreteClass.OrphanGroupStage import OrphanGroupStage


if __name__ == "__main__":

    # set up
    config = JsonConfig("data/config.json")
    config.setup_logger()
    maskGenerator = MrcnnMaskGenerator(config)
    keypointsGenerator = SiftKeypointsGenerator(config)

    # set up group stages
    matcher = SiftKeypointsMatcher(config)
    group_stages = [
        MetadataGroupStage(config),
        RepGroupStage(config, matcher),
        OrphanGroupStage(config, matcher)
    ]

    # generate mask and keypoints for each image
    for image_path in config.get_image_list():
        kps_path = SiftKeypoints.generate_keypoints_path(config, image_path)
        if not os.path.isfile(kps_path):
            imageObj = SnowLeopardImage(image_path)
            keypointsGenerator.generate_and_save_keypoints(imageObj, kps_path)

    # run through each group stage
    groups = []
    
    for stage in group_stages:
        groups = stage.process(groups)
