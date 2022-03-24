from typing import Union
import torch
import numpy as np


class sigTriggerAttack(object):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    def __init__(self,
                 delta : Union[int, float, complex, np.number, torch.Tensor] = 40,
                 f : Union[int, float, complex, np.number, torch.Tensor] =6
                 ) -> None:

        self.delta = delta
        self.f = f

    def __call__(self, img, target = None, image_serial_id = None):
        return self.sigTrigger(img)


    def sigTrigger(self, img):
        """
        Implement paper:
        > Barni, M., Kallas, K., & Tondi, B. (2019).
        > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
        > arXiv preprint arXiv:1902.11237
        superimposed sinusoidal backdoor signal with default parameters
        """

        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
              for j in range(int(img.shape[1])):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))

        return img


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    img = (imageio.imread('/Users/chenhongrui/Documents/Screen Shot 2021-12-30 at 8.09.07 PM.png')[:,:,[0,1,2]])#np.zeros((32,32,3))

    for i in range(0,7):
        trans = sigTriggerAttack(
            alpha = 0.2,
            delta = 20,
            f = i,
        )
        plt.title(f"{i}")
        plt.imshow(trans(img))
        plt.show()


    #
    # plt.imshow(
    #     trans(
    #     np.stack([img, 1-img],0)
    # )[0]
    # )
    # plt.show()
    #
    # plt.imshow(
    #     trans(
    #         np.stack([img, 1 - img],0)
    #     )[1]
    # )
    # plt.show()







