# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
sys.path.append('../../')
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.blended import blendedImageAttack
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_img_transform.ftrojann import ftrojann_version
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize
from utils.bd_img_transform.ctrl import ctrl


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)
npToFloat32 = convertNumpyArrayToFloat32()

class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)
npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack in ['badnet',]:


        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(args.patch_mask_path)),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack == 'blended':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha)), True), # 0.1,
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)), True), # 0.1,
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack == 'sig':
        trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack in ['SSBA']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
    elif args.attack in ['label_consistent']:
        add_trigger = labelConsistentAttack(reduced_amplitude=args.reduced_amplitude)
        add_trigger_func = add_trigger.poison_from_indices
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            # (SSBA_attack_replace_version(
            #     replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            # ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack == 'lowFrequency':

        triggerArray = np.load(args.lowFrequencyPatternPath)

        if len(triggerArray.shape) == 4:
            logging.info("Get lowFrequency trigger with 4 dimension, take the first one")
            triggerArray = triggerArray[0]
        elif len(triggerArray.shape) == 3:
            pass
        elif len(triggerArray.shape) == 2:
            triggerArray =  np.stack((triggerArray,)*3, axis=-1)
        else:
            raise ValueError("lowFrequency trigger shape error, should be either 2 or 3 or 4")

        logging.info("Load lowFrequency trigger with shape {}".format(triggerArray.shape))

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif args.attack == "ctrl":
        train_bd_transform = ctrl(args, train=True)
        test_bd_transform = ctrl(args, train=False)

    elif args.attack == "ftrojann":
        bd_transform = ftrojann_version(YUV=args.YUV, channel_list=args.channel_list, window_size=args.window_size, magnitude=args.magnitude, pos_list=args.pos_list)

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount), int(args.num_classes)
        )

    return bd_label_transform
