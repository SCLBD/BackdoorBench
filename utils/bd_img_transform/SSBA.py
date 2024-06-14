# This script is for SSBA,
# but the main part please refer to https://github.com/tancik/StegaStamp and follow the original paper of SSBA.
# This script only use to replace the img after backdoor modification.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger

from typing import Sequence
import logging
import numpy as np


class SSBA_attack_replace_version(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, replace_images: Sequence) -> None:
        logging.debug(
            'in SSBA_attack_replace_version, the real transform does not happen here, input img, target must be NONE, only image_serial_id used')
        self.replace_images = replace_images

    def __call__(self, img: None,
                 target: None,
                 image_serial_id: int
                 ) -> np.ndarray:
        return self.replace_images[image_serial_id]