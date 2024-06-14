'''
Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks
git link: https://github.com/DreamtaleCore/Refool

Note that
1. (!!! IMPORTANT !!!)
Since the original code is NOT designed to carry out a target backdoor attack,
we have modified the code to make it compatible with the goal of a targeted backdoor attack.
- Change alpha_t to be fixed at 0.4 since the ASR is quite unstable once it is set to be random and often drops a lot.
- change from clean-label to dirty-label. (Even if all params initially random in blend_images are fixed,
    the ASR cannot reach a high enough value with half of the target class being poisoned under the clean-label setting.)
(For why the original code is not target backdoor attack compatible, I refer to the definition mentioned in
https://github.com/DreamtaleCore/Refool/blob/9948043c70170ffdd77ee10806abe2e52bc4dca6/README.md :
"Please note that the output of this command produces the model's classification success rate r, then the attack success rate should be 1-r.".)

2. Since the source GitHub repo gives the download link of the r_adv images,
we use them to generate the blended images without retraining models to select images.
Moreover, as it says, "...once selected, reflection images in Radv can be directly applied to invade a wide range of datasets..." (in arxiv:2007.02343v2),
So, here, we use the same Radv for all datasets to ensure the reproducibility of our experiment results.

3. We directly adopt the blend_images function from the original code to make sure the implementation of the poisoning dataset is the same.

@inproceedings{Liu2020Refool,
	title={Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks},
	author={Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu},
	booktitle={ECCV},
	year={2020}
}

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense

LICENSE is at the end of this file
'''

import sys, yaml, os, argparse, logging, torch
import numpy as np

sys.path = ["./"] + sys.path

import cv2
import random
import numpy as np
import scipy.stats as st
from PIL import Image
from typing import Union
from copy import deepcopy

from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms as transforms
from attack.badnet import BadNet, add_common_attack_args
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform, get_labels
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate, general_compose


def blend_images(
        img_t: Union[Image.Image, np.ndarray],
        img_r: Union[Image.Image, np.ndarray],
        max_image_size=560,
        ghost_rate=0.49,
        alpha_t=-1., # depth of field, intensity number # negative value means randomly pick (see code below)
        offset=(0, 0), # Ghost effect delta (spatial shift)
        sigma=-1, # out of focus sigma # negative value means randomly pick (see code below)
        ghost_alpha=-1. # Ghost effect alpha # negative value means randomly pick (see code below)
    ) -> (np.ndarray, np.ndarray, np.ndarray): # all np.uint8
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image


    return blended, transmission_layer, reflection_layer
    all return value is np array in uint8

    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.random() < ghost_rate:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :],
                             (w, h), cv2.INTER_CUBIC)
        reflection_mask = ghost_r * (1 - alpha_t)

        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.clip(np.power(reflection_mask, 1 / 2.2), 0, 1)
        blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        # generate the blended image with focal blur
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[:2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = gen_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer

# a = Image.fromarray(np.zeros((32,32,3)).astype(np.uint8))
# b = Image.fromarray(np.ones((32,32,3)).astype(np.uint8))
#
# blended, transmission_layer, reflection_layer = blend_images(
#         a,
#         b,
#         max_image_size=560,
#         ghost_rate=0.49,
#         alpha_t=-1.,
#         offset=(0, 0),
#         sigma=-1,
#         ghost_alpha=-1.
#     )
# print((blended.dtype))
# print((transmission_layer.dtype))
# print((reflection_layer.dtype))

class RefoolTrigger(object):


    def __init__(self,
                 R_adv_pil_img_list,
                 img_height,
                 img_width,
                 ghost_rate,
                 alpha_t=-1.,  # depth of field, intensity number # negative value means randomly pick (see code below)
                 offset=(0, 0),  # Ghost effect delta (spatial shift)
                 sigma=-1,  # out of focus sigma # negative value means randomly pick (see code below)
                 ghost_alpha=-1.  # Ghost effect alpha # negative value means randomly pick (see code below)
                 ):
        '''

        :param R_adv: PIL image list
        '''

        self.R_adv_pil_img_list = R_adv_pil_img_list
        self.img_height = img_height
        self.img_width = img_width
        self.ghost_rate = ghost_rate
        self.alpha_t = alpha_t
        self.offset = offset
        self.sigma = sigma
        self.ghost_alpha = ghost_alpha

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)
    def add_trigger(self, img):
        reflection_pil_img = self.R_adv_pil_img_list[np.random.choice(list(range(len(self.R_adv_pil_img_list))))]
        return blend_images(
            img,
            reflection_pil_img,
            max_image_size = max(self.img_height,self.img_width),
            ghost_rate = self.ghost_rate,
            alpha_t = self.alpha_t,
            offset = self.offset,
            sigma = self.sigma,
            ghost_alpha = self.ghost_alpha,
        )[0] # we need only the blended img

class Refool(BadNet):
    r"""Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks

        basic structure:

        1. config args, save_path, fix random seed
        2. set the clean train data and clean test data
        3. set the attack img transform and label transform
        4. set the backdoor attack data and backdoor test data
        5. set the device, model, criterion, optimizer, training schedule.
        6. save the attack result for defense

        .. code-block:: python
            attack = Refool()
            attack.attack()

        .. Note::
            @inproceedings{Liu2020Refool,
                title={Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks},
                author={Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu},
                booktitle={ECCV},
                year={2020}
            }
        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (Int): target class No. in all2one attack
            attack_label_trans (str): which type of label modification in backdoor attack
            pratio (float): the poison rate
            bd_yaml_path (string): path for yaml file provide additional default attributes
            r_adv_img_folder_path(float): where the selected r_adv put, used for generate blended imgs
            ghost_rate(float): ghost rate for blended imgs
            alpha_t(float): depth of field, intensity number (negative value means randomly pick)
            offset(int): Ghost effect delta (spatial shift)
            sigma(float): out of focus sigma (negative value means randomly pick)
            ghost_alpha(float): Ghost effect alpha (negative value means randomly pick)
            **kwargs (optional): Additional attributes.

        """

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

        parser = add_common_attack_args(parser)
        parser.add_argument("--r_adv_img_folder_path", type = float, help = "where the selected r_adv put, used for generate blended imgs")
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/refool/default.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument('--ghost_rate', type=float,
                            help='ghost rate for blended imgs')
        parser.add_argument('--alpha_t', type = float, help = "depth of field, intensity number # negative value means randomly pick") # alpha_t=-1.,  #
        parser.add_argument("--offset", type=int, nargs='+',help = "Ghost effect delta (spatial shift)") # offset=(0, 0)
        parser.add_argument('--sigma',type = float,help = "out of focus sigma # negative value means randomly pick") # sigma=-1,  # out
        parser.add_argument('--ghost_alpha',type = float, help = "Ghost effect alpha # negative value means randomly pick") # ghost_alpha=-1.  # Ghost
        parser.add_argument("--clean_label", type = int, )
        return parser

    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        reflection_img_list = []
        for img_name in os.listdir(args.r_adv_img_folder_path):
            full_img_path = os.path.join(args.r_adv_img_folder_path, img_name)
            reflection_img = Image.open(full_img_path)
            reflection_img_list.append(
                trans(reflection_img)
            )
            reflection_img.close()

        bd_transform = RefoolTrigger(
            reflection_img_list,
            args.img_size[0],
            args.img_size[1],
            args.ghost_rate,
            alpha_t = args.alpha_t,
            offset = args.offset,
            sigma = args.sigma,
            ghost_alpha = args.ghost_alpha,
        )

        train_bd_img_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

        test_bd_img_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)

        ### 4. set the backdoor attack data and backdoor test data
        clean_label = True if args.clean_label == 1 else False
        logging.info(f"clean_label set to {clean_label}")
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
            clean_label=clean_label,
        )

        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle',
                   )

        ### generate train dataset for backdoor attack
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform


if __name__ == '__main__':
    attack = Refool()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

'''
LICENSE
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.


Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More_considerations
     for the public: 
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution-NonCommercial 4.0 International Public
License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution-NonCommercial 4.0 International Public License ("Public
License"). To the extent this Public License may be interpreted as a
contract, You are granted the Licensed Rights in consideration of Your
acceptance of these terms and conditions, and the Licensor grants You
such rights in consideration of benefits the Licensor receives from
making the Licensed Material available under these terms and
conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.
  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. NonCommercial means not primarily intended for or directed towards
     commercial advantage or monetary compensation. For purposes of
     this Public License, the exchange of the Licensed Material for
     other material subject to Copyright and Similar Rights by digital
     file-sharing or similar means is NonCommercial provided there is
     no payment of monetary compensation in connection with the
     exchange.

  j. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  k. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  l. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part, for NonCommercial purposes only; and

            b. produce, reproduce, and Share Adapted Material for
               NonCommercial purposes only.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties, including when
          the Licensed Material is used other than for NonCommercial
          purposes.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database for NonCommercial purposes
     only;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the "Licensor." The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
'''