'''
@article{saha2019hidden,
  title={Hidden Trigger Backdoor Attacks},
  author={Saha, Aniruddha and Subramanya, Akshayvarun and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:1910.00033},
  year={2019}
}

rewrite from : https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
'''

import sys, yaml, os
import sys, os, argparse, torch
from pprint import pformat
import numpy as np
from utils.aggregate_block.save_path_generate import generate_save_folder
import time
import logging
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform, generate_single_target_attack_train_pidx
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.bd_groupwise_transform.groupwise_feature_disguise_pgd_perturbation import groupwise_feature_disguise_pgd_perturbation
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

# TODO add the default setting to yaml file.

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--mode', type=str,
    #                     help='classification/detection/segmentation')
    parser.add_argument('--pretrained_model_path', type = str,
                        help= 'the pretrained model path')
    parser.add_argument('--eps1', type = float,
                        help = 'for pgd in poison data generation')
    parser.add_argument('--pgd_init_lr', type = float,
                        help='for pgd in poison data generation')
    parser.add_argument('--pgd_max_iter', type = int,
                        help='for pgd in poison data generation')

    parser.add_argument('--yaml_path', type=str, default='../config/HiddenTriggerBackdoorAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')

    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--attack', type = str, help = 'the attack used in hiddentrigger')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--img_size', type=list)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')

    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')

    return parser

def main():

    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv



    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')



    # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    try:
        import wandb
        wandb.init(
            project="bdzoo2",
            entity="chr",
            name=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + os.path.basename(save_path),
            config=args,
        )
        set_wandb = True
    except:
        set_wandb = False

    logging.info(f'set_wandb = {set_wandb}')


    fix_random(int(args.random_seed))



    train_dataset_without_transform, \
                train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)




    benign_train_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        )

    benign_train_dl = DataLoader(
        benign_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    benign_test_dl = DataLoader(
        prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )



    # put it here, since the perturbation need the pretrained model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")



    net = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )

    trainer = generate_cls_trainer(
        model = net,
        attack_name=args.attack,
    )

    net.load_state_dict(torch.load(args.pretrained_model_path))




    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)

    # Notice that for CLEAN LABEL attack, this blocks only return the label_trans for TEST time !!!
    bd_label_transform = bd_attack_label_trans_generate(args)




    # start poison injection
    target_train_ds = deepcopy(benign_train_ds)
    target_train_ds.subset(np.where(benign_train_ds.original_targets == args.attack_target)[0])
    source_train_ds = deepcopy(benign_train_ds)
    source_train_ds.subset(np.where(benign_train_ds.original_targets != args.attack_target)[0])

    train_loader_target = torch.utils.data.DataLoader(target_train_ds,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        # num_workers=8,
                                                        # pin_memory=True
                                                      )
    iter_target = iter(train_loader_target)

    train_loader_source = torch.utils.data.DataLoader(source_train_ds,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True,
                                                      # num_workers=8,
                                                      # pin_memory=True
                                                      )
    iter_source = iter(train_loader_source)

    pnum =  args.pratio * len(benign_train_ds) if 'pratio' in args.__dict__ else args.p_num

    disguised_img_all = []
    disguise_img_index_all = []
    poison_count = 0

    train_bd_img_transform.target_image = torch.tensor(np.transpose(train_bd_img_transform.target_image,(2,0,1)))

    for _ in range(len(train_loader_target)):

        # LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
        #img, label, self.original_index[item], self.poison_indicator[item], self.original_targets[item],
        source_img, source_label, source_original_index, _ , _  = next(iter_source)
        target_img, target_label, target_original_index, _ , _  = next(iter_target)

        # add patch trigger for each photo
        for source_img_i in range(len(source_img)):
            source_img[source_img_i] = train_bd_img_transform(source_img[source_img_i], source_label[source_img_i], source_original_index[source_img_i])

        disguised_img = groupwise_feature_disguise_pgd_perturbation(
            patched_source_img_batch = source_img,
            target_img_batch = target_img,
            model = net,
            device = device,
            img_eps1 = args.eps1,
            pgd_init_lr = args.pgd_init_lr,
            pgd_max_iter = args.pgd_max_iter,
        )

        poison_count += args.batch_size
        disguised_img_all.append(disguised_img)
        disguise_img_index_all.append(target_label)

        if  poison_count >= pnum:
            disguised_img_all = torch.cat(disguised_img_all)
            disguise_img_index_all = torch.cat(disguise_img_index_all)
            break

    if  poison_count < pnum:
        sys.exit("cannot generate enough poison samples, plz check pgd setting or poison num is set being impossible")

    adv_train_ds =  prepro_cls_DatasetBD(
        deepcopy(train_dataset_without_transform),
        poison_idx= np.zeros(len(train_dataset_without_transform)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
    )

    for iter_i in range(pnum): # only pnum of target class being changed
        adv_train_ds.data[disguise_img_index_all[iter_i]] = disguised_img_all[iter_i]
        # clean-label only, so only img change. No need to match index, since adv_train_ds still in order
        adv_train_ds.poison_indicator[disguise_img_index_all[iter_i]] = 1

    adv_train_dl = DataLoader(
        dataset = adv_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_pidx = generate_pidx_from_label_transform(
        benign_test_dl.dataset.targets,
        label_transform=bd_label_transform,
        train=False,
    )

    adv_test_dataset = prepro_cls_DatasetBD(
        deepcopy(test_dataset_without_transform),
        poison_idx=test_pidx,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=test_img_transform,
        ori_label_transform_in_loading=test_label_transform,
        add_details_in_preprocess=True,
    )

    adv_test_dataset.subset(
        np.where(test_pidx == 1)[0]
    )

    adv_test_dl = DataLoader(
        dataset = adv_test_dataset,
        batch_size= args.batch_size,
        shuffle= False,
        drop_last= False,
    )



    criterion = argparser_criterion(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)


    if __name__ == '__main__':

        if 'load_path' not in args.__dict__:

            trainer.train_with_test_each_epoch(
                train_data = adv_train_dl,
                test_data = benign_test_dl,
                adv_test_data = adv_test_dl,
                end_epoch_num = args.epochs,
                criterion = criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device,
                frequency_save = args.frequency_save,
                save_folder_path = save_path,
                save_prefix = 'attack',
                continue_training_path = None,
            )

        else:

            if 'recover' not in args.__dict__ or args.recover == False :

                print('finetune so use less data, 5% of benign train data')

                benign_train_dl.dataset.subset(
                    np.random.choice(
                        np.arange(
                            len(benign_train_dl.dataset)),
                        size=round((len(benign_train_dl.dataset)) / 20),  # 0.05
                        replace=False,
                    )
                )

                trainer.train_with_test_each_epoch(
                    train_data=benign_train_dl,
                    test_data=benign_test_dl,
                    adv_test_data=adv_test_dl,
                    end_epoch_num=args.epochs,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    frequency_save=args.frequency_save,
                    save_folder_path=save_path,
                    save_prefix='finetune',
                    continue_training_path=args.load_path,
                    only_load_model=True,
                )

            elif 'recover' in args.__dict__ and args.recover == True :

                trainer.train_with_test_each_epoch(
                    train_data=adv_train_dl,
                    test_data=benign_test_dl,
                    adv_test_data=adv_test_dl,
                    end_epoch_num=args.epochs,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    frequency_save=args.frequency_save,
                    save_folder_path=save_path,
                    save_prefix='attack',
                    continue_training_path=args.load_path,
                    only_load_model=False,
                )



    save_attack_result(
        model_name = args.model,
        num_classes = args.num_classes,
        model = trainer.model.cpu().state_dict(),
        data_path = args.dataset_path,
        img_size = args.img_size,
        clean_data = args.dataset,
        bd_train = adv_train_ds,
        bd_test = adv_test_dataset,
        save_path = save_path,
    )

if __name__ == '__main__':
    main()