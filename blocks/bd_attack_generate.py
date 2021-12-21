import sys, logging
sys.path.append('../')
import torch, imageio
import torchvision.transforms as transforms


from tool.backdoor_img_transform import *
from tool.backdoor_label_transform import *

def bd_attack_img_trans_generate(args):

    if args.attack == 'fix_patch':

        trigger_loc = args.attack_trigger_loc # [[26, 26], [26, 27], [27, 26], [27, 27]]
        trigger_ptn = args.trigger_ptn # torch.randint(0, 256, [len(trigger_loc)])
        bd_transform = AddPatchTrigger(
            trigger_loc=trigger_loc,
            trigger_ptn=trigger_ptn,
        )
        train_bd_transform = bd_transform
        test_bd_transform = bd_transform

    elif args.attack == 'blended':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha)) # 0.1
        test_bd_transform = blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)) # 0.1

    elif args.attack == 'input':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = inputInstanceKeyAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).numpy().transpose(1, 2, 0) * 255,
            args.attack_train_pixel_perturb_limit)#5
        test_bd_transform = inputInstanceKeyAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
            ).numpy().transpose(1, 2, 0) * 255,
            args.attack_test_pixel_perturb_limit)#5

    elif args.attack == 'SSBA_replace':
        # replace just remind you that this function can only use the SSBA attack img to do replace, NOT do transform
        from PIL import Image
        train_bd_transform = SSBA_attack_replace_version(
            replace_images=np.load(args.attack_train_replace_imgs_path) # '../data/cifar10_SSBA/train.npy'
        )
        test_bd_transform = SSBA_attack_replace_version(
            replace_images=np.load(args.attack_test_replace_imgs_path) #'../data/cifar10_SSBA/test.npy'
        )

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    a = AllToOne_attack(target_label=4)
    b = AllToAll_shiftLabelAttack(2, 10)
    c = OneToAll_randomLabelAttack(3, [4,5,6])
    d = OneToOne_attack(3,4)
    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)  # random.choice([i for i in range(10) if i != source_label])
        bd_label_transform = AllToOne_attack(target_label)  # OneToOne_attack(source_label, target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(args.attack_label_shift_amount), int(args.num_classses)
        )
    elif args.attack_label_trans == 'one2all':
        bd_label_transform = OneToAll_randomLabelAttack(
            int(args.attack_source_label),
            list(args.attack_target),
        )
    elif args.attack_label_trans == 'one2one':
        bd_label_transform = OneToOne_attack(
            int(args.attack_source_label),
            int(args.attack_target)
        )
    return bd_label_transform

