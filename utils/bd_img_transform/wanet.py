'''
rewrite from
    @inproceedings{
    nguyen2021wanet,
    title={WaNet - Imperceptible Warping-based Backdoor Attack},
    author={Tuan Anh Nguyen and Anh Tuan Tran},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=eEn8KTtJOx}
    }
    code : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
'''


import torch
import numpy as np
import torch.nn.functional as F

def npFloatImgUint8ImgSwitch(
        img : np.ndarray,
       ):
    if img.dtype == np.dtype('uint8'):
        return np.float32(img) / 255.
    elif img.dtype in [
        np.dtype('float16'),
        np.dtype('float32'),
        np.dtype('float64'),
    ]:
        return np.uint8(img * 255)

def nCHW_to_nHWC(images):
    return images.transpose((0, 2, 3, 1))

def nHWC_to_nCHW(images):
    return images.transpose((0, 3, 1, 2))

class imageWarp(object):

    '''
    The non-add noise mode have the SAME warping matrix after init
    but add random noise WILL CHANGE for each img input

    this class
    tensor in tensor out  (MUST be nCHW)
    np in np out (HWC or nHWC)
    '''


    def __init__(self,
                 warp_kernel_size,
                 image_height,
                 warping_strength,
                 grid_rescale,
                 add_random_noise,
                 device=torch.device('cpu')
                 ):

        '''
        :param warp_kernel_size:
        :param image_height:
        :param warping_strength:
        :param grid_rescale: in order to avoid get out of (-1,1)
        :param add_random_noise: whether to add random noise warp to EACH img
        '''
        self.warp_kernel_size = warp_kernel_size
        self.image_height = image_height
        self.warping_strength = warping_strength
        self.grid_rescale = grid_rescale
        self.add_random_noise = add_random_noise
        self.device = device

        ins = torch.rand(1, 2, warp_kernel_size, warp_kernel_size) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
        ins = ins / torch.mean(
            torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1
        noise_grid = (
            F.interpolate(ins, size=image_height, mode="bicubic",
                       align_corners=True).permute(0, 2, 3, 1)  # here upsample and make the dimension match
        )
        array1d = torch.linspace(-1, 1, steps=image_height)
        x, y = torch.meshgrid(array1d,
                              array1d)  # form two mesh grid correspoding to x, y of each position in height * width matrix
        identity_grid = torch.stack((y, x), 2)[None, ...]  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))

        grid_temps = (identity_grid + warping_strength * noise_grid / image_height) * grid_rescale
        self.grid_temps = torch.clamp(grid_temps, -1, 1).to(device)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def tensor_version_add_trigger(self, img):

        if self.add_random_noise == False:
            img = F.grid_sample(img.to(self.device), self.grid_temps.repeat(img.shape[0], 1, 1, 1), align_corners=True)
        else:
            ins = (torch.rand(img.shape[0], self.image_height, self.image_height, 2) * 2 - 1).to(self.device)
            grid_temps2 = self.grid_temps.repeat(img.shape[0], 1, 1, 1) + ins / self.image_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            img = F.grid_sample(img, grid_temps2, align_corners=True)

        return img

    def add_trigger(self,
                    img,
                    ):

        '''

        :param img: np.array
        :return:
        '''
        if isinstance(img, torch.Tensor):

            img = self.tensor_version_add_trigger(img)

        elif isinstance(img, np.ndarray):

            if img.dtype == np.dtype('uint8'):
                img = npFloatImgUint8ImgSwitch(img)

            if len(img.shape) == 3: #single one
                img = img[None, ...]

            img = torch.tensor(nHWC_to_nCHW(img)).float()

            img = self.tensor_version_add_trigger(img)

            img = nCHW_to_nHWC(img.cpu().numpy())

        return img[0] if img.shape[0] == 1 else img


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    img = (imageio.imread('/Users/chenhongrui/Documents/Screen Shot 2021-12-30 at 8.09.07 PM.png')[:,:,[0,1,2]])#np.zeros((32,32,3))

    trans = imageWarp(
        4,
        32,
        0.5,
        0.98,
        False
    )

    trans2 = imageWarp(
        4,
        32,
        0.5,
        0.98,
        True
    )


    plt.imshow(trans(img))
    plt.show()

    plt.imshow(trans2(img))
    plt.show()

    plt.imshow(
        trans(
        np.stack([img, 1-img],0)
    )[0]
    )
    plt.show()

    plt.imshow(
        trans(
            np.stack([img, 1 - img],0)
        )[1]
    )
    plt.show()

    plt.imshow(
        trans2(
            np.stack([img, 1 - img], 0)
        )[0]
    )
    plt.show()

    plt.imshow(
        trans2(
            np.stack([img, 1 - img], 0)
        )[1]
    )
    plt.show()






