import numpy as np

# note that here we use np array in np with uint8
class inputInstanceKeyAttack(object):
    '''
    from Chen et al. - 2017 - Targeted Backdoor Attacks on Deep Learning Systems
    just replace the original image with attack image + some Gaussian noise.
    '''
    def __init__(self, target_image, pixel_perturb_limit, perturb_func = None):
        self.target_image = target_image

        assert 0<= pixel_perturb_limit <= 255
        self.pixel_perturb_limit = pixel_perturb_limit

        if perturb_func is not None:
            self.perturb_func = perturb_func
        else:
            self.perturb_func = np.random.rand

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger()

    def add_trigger(self):

        return np.clip(
            self.target_image + self.perturb_func(*self.target_image.shape) * self.pixel_perturb_limit,
            a_min = 0,
            a_max = 255)