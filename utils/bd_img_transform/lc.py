'''
original code from
link : https://github.com/MadryLab/label-consistent-backdoor-code
'''
import numpy as np
import logging
from torchvision.transforms import Resize, InterpolationMode
import torch


class labelConsistentAttack(object):

    '''
    This class ONLY add square trigger to the image !!!!!
    This class ONLY add square trigger to the image !!!!!
    This class ONLY add square trigger to the image !!!!!
    For adversarial attack to origianl images part before adding trigger, plz refer to resource/label-consistent folder for more details.
    '''

    def __init__(self, trigger = "all-corners", reduced_amplitude=1.0):

        assert 0 <= reduced_amplitude <= 1, "reduced_amplitude is in [0,1] !"
        logging.warning("Original code only give trigger in 32 * 32. For other image size, we do resize to the mask with InterpolationMode.NEAREST. \nIf you do not agree with our implememntation, you can rewrite utils/bd_img_transform/lc.py in your own way.")
        logging.info(f"For Label-consistent attack, reduced_amplitude (transparency) = {reduced_amplitude}, 0 means no square trigger, 1 means no reduction.")
        if reduced_amplitude == 0:
            logging.warning("!!! reduced_amplitude = 0, note that this mean NO square trigger is added after adversarial attack to origianl image!!!")

        logging.warning(f"You are using pattern: {trigger} for labelConsistentAttack")

        self.trigger_mask = [] # For overriding pixel values
        self.trigger_add_mask = [] # For adding or subtracting to pixel values
        if trigger == "bottom-right":
            self.trigger_mask = [
                ((-1, -1), 1),
                ((-1, -2), -1),
                ((-1, -3), 1),
                ((-2, -1), -1),
                ((-2, -2), 1),
                ((-2, -3), -1),
                ((-3, -1), 1),
                ((-3, -2), -1),
                ((-3, -3), -1)
            ]
        elif trigger == "all-corners":
            self.trigger_mask = [
                ((0, 0), 1),
                ((0, 1), -1),
                ((0, 2), -1),
                ((1, 0), -1),
                ((1, 1), 1),
                ((1, 2), -1),
                ((2, 0), 1),
                ((2, 1), -1),
                ((2, 2), 1),

                ((-1, 0), 1),
                ((-1, 1), -1),
                ((-1, 2), 1),
                ((-2, 0), -1),
                ((-2, 1), 1),
                ((-2, 2), -1),
                ((-3, 0), 1),
                ((-3, 1), -1),
                ((-3, 2), -1),

                ((0, -1), 1),
                ((0, -2), -1),
                ((0, -3), -1),
                ((1, -1), -1),
                ((1, -2), 1),
                ((1, -3), -1),
                ((2, -1), 1),
                ((2, -2), -1),
                ((2, -3), 1),

                ((-1, -1), 1),
                ((-1, -2), -1),
                ((-1, -3), 1),
                ((-2, -1), -1),
                ((-2, -2), 1),
                ((-2, -3), -1),
                ((-3, -1), 1),
                ((-3, -2), -1),
                ((-3, -3), -1),
            ]
        else:
            assert False

        self.reduced_amplitude = reduced_amplitude
        if reduced_amplitude == "none":
            self.reduced_amplitude = None

    def resize_annotation(self, annotation, img_size):
        # eg. list of ((-3, -3), -1),

        if (img_size == (32,32)) or (len(annotation) == 0):
            return annotation

        mask = np.zeros((32,32))
        for (x, y), value in annotation:
            mask[x][y] = value

        resize = Resize(img_size, interpolation=InterpolationMode.NEAREST)
        resized_mask = resize(torch.from_numpy(mask)[None,...])[0]

        new_annotation = []
        resized_mask = resized_mask.numpy()
        for x,y in zip(np.nonzero(resized_mask)[0].tolist(),np.nonzero(resized_mask)[1].tolist()):
            new_annotation.append(((x,y), resized_mask[x][y]))

        return new_annotation

    def poison_from_indices(self, image, apply_trigger=True):

        max_allowed_pixel_value = 255

        image_new = np.copy(image).astype(np.float32)

        trigger_mask = self.trigger_mask
        trigger_add_mask = self.trigger_add_mask

        if self.reduced_amplitude is not None:
            trigger_add_mask = [
                ((x, y), mask_val * self.reduced_amplitude)
                for (x, y), mask_val in trigger_mask
            ]

            trigger_mask = []

        trigger_mask = [
            ((x, y), max_allowed_pixel_value * value)
            for ((x, y), value) in trigger_mask
        ]
        trigger_add_mask = [
            ((x, y), max_allowed_pixel_value * value)
            for ((x, y), value) in trigger_add_mask
        ]

        if apply_trigger:
            trigger_mask = self.resize_annotation(trigger_mask, image.shape[:2])
            for (x, y), value in trigger_mask:
                image_new[x][y] = value
            trigger_add_mask = self.resize_annotation(trigger_add_mask, image.shape[:2])
            for (x, y), value in trigger_add_mask:
                image_new[x][y] += value

        image_new = np.clip(image_new, 0, max_allowed_pixel_value)

        # debug block
        # print("min image", (image).min())
        # print("max image", (image).max())
        # print("min image_new", (image_new).min())
        # print("max image_new", (image_new).max())
        # print("image_new - image", image_new - image)
        # print("sum image_new - image", image_new - image)

        return image_new

if __name__ == '__main__':
    # test
    for trigger in ["bottom-right","all-corners"]:
        for reduced_amplitude in [0,1,0.5,1]:
            a = labelConsistentAttack(trigger, reduced_amplitude)
            a.poison_from_indices(np.zeros((32,32,3)))
            a.poison_from_indices(np.zeros((64, 32, 3)))
            a.poison_from_indices(np.zeros((64, 64, 3)))
