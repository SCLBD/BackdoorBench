



class SSBA_attack_replace_version(object):
    '''in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here'''

    def __init__(self, replace_images: Sequence) -> None:
        logging.info(
            'in SSBA_attack_replace_version, the real transform does not happen here, input img, target must be NONE, only image_serial_id used')
        self.replace_images = replace_images

    def __call__(self, img: None,
                 target: None,
                 image_serial_id: int
                 ) -> np.ndarray:
        return self.replace_images[image_serial_id]