'''
Targeted backdoor attacks on deep learning systems using data poisoning
this script is for blended attack

@article{Blended,
	title        = {Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning},
	author       = {Xinyun Chen and Chang Liu and Bo Li and Kimberly Lu and Dawn Song},
	journal      = {arXiv preprint arXiv:1712.05526},
	year         = {2017}
}
basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense

'''
import argparse
import os
import sys

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args


class Blended(BadNet):
    r'''Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning

        basic structure:

        1. config args, save_path, fix random seed
        2. set the clean train data and clean test data
        3. set the attack img transform and label transform
        4. set the backdoor attack data and backdoor test data
        5. set the device, model, criterion, optimizer, training schedule.
        6. attack or use the model to do finetune with 5% clean data
        7. save the attack result for defense

        .. code-block:: python

            attack = Blended()
            attack.attack()

        .. Note::
            @article{Blended,
                title        = {Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning},
                author       = {Xinyun Chen and Chang Liu and Bo Li and Kimberly Lu and Dawn Song},
                journal      = {arXiv preprint arXiv:1712.05526},
                year         = {2017}
            }

        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (Int): target class No. in all2one attack
            attack_label_trans (str): which type of label modification in backdoor attack
            pratio (float): the poison rate
            bd_yaml_path (string): path for yaml file provide additional default attributes
            attack_trigger_img_path (string): path for trigger image
            attack_train_blended_alpha (float): alpha for blended attack, for train dataset
            attack_test_blended_alpha  (float): alpha for blended attack, for test dataset
            **kwargs (optional): Additional attributes.

        '''

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--attack_trigger_img_path", type=str, )
        parser.add_argument("--attack_train_blended_alpha", type=float, )
        parser.add_argument("--attack_test_blended_alpha", type=float, )
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/blended/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser


if __name__ == '__main__':
    attack = Blended()
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
