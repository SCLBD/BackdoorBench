'''
Badnets: Identifying vulnerabilities in the machine learning model supply chain.
this script is for badnet attack

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense


'''

import os
import sys
import yaml

sys.path = ["./"] + sys.path

import argparse
import numpy as np
import torch
import logging

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform


def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    return parser


class BadNet(NormalCase):
    r"""Badnets: Identifying vulnerabilities in the machine learning model supply chain.

        basic structure:

        1. config args, save_path, fix random seed
        2. set the clean train data and clean test data
        3. set the attack img transform and label transform
        4. set the backdoor attack data and backdoor test data
        5. set the device, model, criterion, optimizer, training schedule.
        6. attack or use the model to do finetune with 5% clean data
        7. save the attack result for defense

        .. code-block:: python
            attack = BadNet()
            attack.attack()

        .. Note::
            @article{gu2017badnets,
            title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
            author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
            journal={arXiv preprint arXiv:1708.06733},
            year={2017}}

        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (Int): target class No. in all2one attack
            attack_label_trans (str): which type of label modification in backdoor attack
            pratio (float): the poison rate
            bd_yaml_path (string): path for yaml file provide additional default attributes
            patch_mask_path (string): path for patch mask
            **kwargs (optional): Additional attributes.

        """

    def __init__(self):
        super(BadNet).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        parser.add_argument("--patch_mask_path", type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/badnet/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, 'r') as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults

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

        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)

        ### 4. set the backdoor attack data and backdoor test data
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
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

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


if __name__ == '__main__':
    attack = BadNet()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
