class blendedImageAttack(object):

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image
