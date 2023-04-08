# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
sys.path.append('../../')
import  imageio
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.blended import blendedImageAttack
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize

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

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack == 'fix_patch':

        # trigger_loc = args.attack_trigger_loc # [[26, 26], [26, 27], [27, 26], [27, 27]]
        # trigger_ptn = args.trigger_ptn # torch.randint(0, 256, [len(trigger_loc)])
        # bd_transform = AddPatchTrigger(
        #     trigger_loc=trigger_loc,
        #     trigger_ptn=trigger_ptn,
        # )

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(np.load(args.patch_mask_path)),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

    elif args.attack == 'blended':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha)), True) # 0.1,
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)), True) # 0.1,
        ])
        
    elif args.attack == 'ft_trojan':
    
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
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
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
        ])

    elif args.attack in ['SSBA_replace']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_train_replace_imgs_path) # '../data/cifar10_SSBA/train.npy'
                ), True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_test_replace_imgs_path) #'../data/cifar10_SSBA/test.npy'
                ),True),
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
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (add_trigger_func, False),
        ])

    elif args.attack == 'lowFrequency':
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = np.load(args.lowFrequencyPatternPath)
            ), True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = np.load(args.lowFrequencyPatternPath)
            ), True),
        ])


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

