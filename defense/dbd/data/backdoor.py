import numpy as np
from PIL import Image


class BadNets(object):
    """BadNets Injection Strategy.

    Reference:
    [1] "Badnets: Evaluating backdooring attacks on deep neural networks."
    Tianyu Gu, et al. IEEE Access 2019.

    Args:
        trigger_path (string): The trigger path.
    
    .. note:: 
      The trigger image specified by the trigger path whose background is in black.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")
        self.trigger_ptn = np.array(trigger_ptn)
        # Get the trigger location since the background is in black
        # and the trigger is in color.
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """Add `trigger_ptn` to `img`.

        Args:
            img (np.ndarray): The input image (HWC).
        
        Returns:
            poison_img (np.ndarray): The poisoned image (HWC).
        """
        img[self.trigger_loc] = 0
        poison_img = img + self.trigger_ptn

        return poison_img


class Blend(object):
    """Blended Injection Strategy.

    Reference:
    [1] "Targeted backdoor attacks on deep learning systems using data poisoning." 
    Xinyun Chen, et al. arXiv:1712.05526.

    Args:
        trigger_path (string): Trigger path.
        alpha (float): The interpolation factor.
    """

    def __init__(self, trigger_path, alpha=0.1):
        with open(trigger_path, "rb") as f:
            self.trigger_ptn = Image.open(f).convert("RGB")
        self.alpha = alpha

    def __call__(self, img):
        return self.blend_trigger(img)

    def blend_trigger(self, img):
        """Blend the input `img` with the `trigger_ptn` by
        alpha * trigger_ptn + (1 - alpha) * img.

        Args:
            img (numpy.ndarray): The input image (HWC).
            
        Return:
            poison_img (np.ndarray): The poisoned image (HWC).
        """
        img = Image.fromarray(img)
        trigger_ptn = self.trigger_ptn.resize(img.size)
        poison_img = Image.blend(img, trigger_ptn, self.alpha)

        return np.array(poison_img)
