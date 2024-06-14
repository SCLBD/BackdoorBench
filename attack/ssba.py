'''
Invisible backdoor attack with sample-specific triggers
this script is for SSBA attack

code link: https://github.com/SCLBD/ISSBA

@inproceedings{ssba,
  title={Invisible backdoor attack with sample-specific triggers},
  author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}

Note that the autoencoder training process and img process part are not in this script,
    which are time comsume and dataset-dependent, please follow https://github.com/tancik/StegaStamp to train models for generating the poisoned data.
    (Or you can find a torch version to generate the poisoned data in ./resource/ssba, please follow the readme in ./resource/ssba)
    Then place the poisoned image array to `attack_train_replace_imgs_path` and `attack_test_replace_imgs_path`

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense


'''

import argparse
import logging
import os
import sys

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape


class SSBA(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

        parser = add_common_attack_args(parser)
        parser.add_argument('--attack_train_replace_imgs_path', type=str)
        parser.add_argument('--attack_test_replace_imgs_path', type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/ssba/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def process_args(self, args):

        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        if ('attack_train_replace_imgs_path' not in args.__dict__) or (args.attack_train_replace_imgs_path is None):
            args.attack_train_replace_imgs_path = f"./resource/ssba/{args.dataset}_ssba_train_b1.npy"
            logging.info(
                f"args.attack_train_replace_imgs_path does not found, so = {args.attack_train_replace_imgs_path}")

        if ('attack_test_replace_imgs_path' not in args.__dict__) or (args.attack_test_replace_imgs_path is None):
            args.attack_test_replace_imgs_path = f"./resource/ssba/{args.dataset}_ssba_test_b1.npy"
            logging.info(
                f"args.attack_test_replace_imgs_path does not found, so = {args.attack_test_replace_imgs_path}")

        return args


if __name__ == '__main__':
    attack = SSBA()
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
