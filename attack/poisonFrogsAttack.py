
'''
@ARTICLE{2018arXiv180400792S,
   author = {{Shafahi}, A. and {Ronny Huang}, W. and {Najibi}, M. and {Suciu}, O. and
	{Studer}, C. and {Dumitras}, T. and {Goldstein}, T.},
    title = "{Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1804.00792},
 primaryClass = "cs.LG",
 keywords = {Computer Science - Learning, Computer Science - Cryptography and Security, Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
     year = 2018,
    month = apr,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180400792S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
code : https://github.com/ashafahi/inceptionv3-transferLearn-poison
'''

import sys, yaml, os
import argparse
from pprint import pprint, pformat
import numpy as np
import torch
from utils.hook_forward_lastHiddenLayerActivationExtractor import lastHiddenActivationExtractor
from utils.aggregate_block.save_path_generate import generate_save_folder
import time
import logging
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader
from utils.backdoor_generate_pindex import generate_single_target_attack_train_pidx
from copy import deepcopy
from torch.utils.data.dataset import TensorDataset
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

def disguise_embeddings(
        source_class_img_tensor : torch.Tensor, # (n, 3, x, x)
        target_class_img_tensor : torch.Tensor, # (n, 3, x, x)
        net : torch.nn.Module,
        device : torch.device,
        max_iter : int, # 200
        terminal_loss : float, # 1e-10
        lr : float,
        beta : float, # 0.25
        final_blended_rate : float, # 0.3 ,negative means NO
):
    '''

    Note that source_class_img_tensor, target_class_img_tensor is actually one-to-one pairwise optimized
    So if you want one-to-many, then just copy the img tensor for all n in first dimension.

    :param source_class_img_tensor:
    :param target_class_img_tensor:
    :param net:
    :param device:
    :param max_iter:
    :param lr:
    :param beta:
    :param final_blended_rate:
    :return:
    '''
    source_class_img_tensor = source_class_img_tensor.to(device)
    source_class_img_tensor.requires_grad = False

    target_class_img_tensor_init_save = target_class_img_tensor.detach().clone()
    target_class_img_tensor_init_save = target_class_img_tensor_init_save.to(device)
    target_class_img_tensor_init_save.requires_grad = False

    target_class_img_tensor = target_class_img_tensor.to(device)
    target_class_img_tensor = target_class_img_tensor.requires_grad_()

    extractor = lastHiddenActivationExtractor(net)

    for i in range(max_iter):

        net.eval()
        net.to(device)

        source_class_img_feature, _ = extractor.generateLastLinearInput(source_class_img_tensor, device)
        target_class_img_feature, _ = extractor.generateLastLinearInput(target_class_img_tensor, device)

        feature_loss = ((source_class_img_feature - target_class_img_feature) ** 2).sum()

        # forward
        target_class_img_tensor = target_class_img_tensor - lr * torch.autograd.grad(feature_loss, inputs= target_class_img_tensor, create_graph= False)[0]

        # backward
        target_class_img_tensor = (target_class_img_tensor + lr * beta * target_class_img_tensor_init_save)/(1 + beta * lr)

        target_class_img_tensor = torch.clamp(target_class_img_tensor, 0, 1).data

        target_class_img_tensor = target_class_img_tensor.to(device)
        target_class_img_tensor = target_class_img_tensor.requires_grad_()
        if target_class_img_tensor.grad is not None:
            target_class_img_tensor.grad.zero_()

        if terminal_loss > feature_loss.item():

            break

    if final_blended_rate > 0:

        target_class_img_tensor = source_class_img_tensor * final_blended_rate + target_class_img_tensor * (1-final_blended_rate)

    return target_class_img_tensor.data

# if __name__ == '__main__':

#     net = resnet18()
#     pic_tensor = disguise_embeddings(
#         source_class_img_tensor = torch.zeros(1,3,32,32),
#         target_class_img_tensor = torch.ones(1,3,32,32),
#         net = net,
#         device = torch.device('cpu'),
#         max_iter = 200,
#         terminal_loss = 1e-10 ,
#         lr = 0.01,
#         beta = 0.25,
#         final_blended_rate = 0.3,
#     )

#     plt.imshow(pic_tensor[0].detach().numpy().transpose(1,2,0))
#     plt.show()

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit

    """
    parser.add_argument(
        '--pretrained_model_path', type = str, help = 'the target model for attack'
    )
    parser.add_argument(
        '--target_instance_index', type= int, help = 'which sample as target sample to misclassify'
    )
    parser.add_argument(
        '--generate_max_iter', type= int, help = 'max iters in generate one poison sample'
    )
    parser.add_argument(
        '--generate_terminal_loss', type=float, help = 'loss threshold in generate one poison sample'
    )
    parser.add_argument(
        '--generate_lr', type=float, help = 'lr in generate one poison sample'
    )
    parser.add_argument(
        '--generate_beta', type=float, help = ' tuned to make the poison instance look realistic in input space'
    )
    parser.add_argument(
        '--generate_final_blended_rate', type=float,
        help = 'final blended ratio, \
               in order to keep the feature overlap of poison sample and target instance. \
               Negative value means do not apply blended at all.'
    )



    parser.add_argument('--yaml_path', type=str, default='../config/poisonFrogsAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')

    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--attack_label_trans', type = str,
        help = 'which type of label modification in backdoor attack'
    )
    parser.add_argument('--pratio', type = float,
        help = 'the poison rate '
    )
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type = str,
                        help = 'which dataset to use'
    )
    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--img_size', type=list)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help = 'weight decay of sgd')

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

    args.attack = 'poisonFrogs'

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



    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    net = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )

    net.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))



    train_dataset_without_transform, \
                train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)




    benign_train_dl = DataLoader(
        prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        ),
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





    train_pidx = generate_single_target_attack_train_pidx(
        targets = benign_train_dl.dataset.targets,
        tlabel = args.attack_target,
        pratio= args.pratio if 'pratio' in args.__dict__ else None,
        p_num= args.p_num if 'p_num' in args.__dict__ else None,
        clean_label = True,
        train=True,
    )
    pnum = round(args.pratio * len(benign_train_dl.dataset.targets)) if 'pratio' in args.__dict__  \
        else (args.p_num if 'p_num' in args.__dict__ else None)
    assert pnum is not None

    train_pidx = np.zeros(len(benign_train_dl.dataset))
    train_pidx[np.random.choice(
        np.where(benign_train_dl.dataset.targets == args.attack_target)[0],
        pnum,
        replace=False
    )] = 1

    torch.save(train_pidx,
        args.save_path + '/train_pidex_list.pickle',
    )



    class poisonFrogs(object):

        def __init__(self,
                     target_instance,
                     net: torch.nn.Module,
                     device: torch.device,
                     max_iter: int,  # 200
                     terminal_loss: float,  # 1e-10
                     lr: float,
                     beta: float,  # 0.25
                     final_blended_rate: float,  # 0.3 ,negative means NO
                     ):

            self.net = net
            self.device = device
            self.max_iter = max_iter
            self.terminal_loss = terminal_loss
            self.lr = lr
            self.beta = beta
            self.final_blended_rate = final_blended_rate

            self.target_instance = target_instance
            self.target_instance_tensor = torch.tensor((np.array(target_instance) / 255).transpose(2,0,1)).float()[None,...]
            #(1,3,x,x)

        def __call__(self, img, target=None, image_serial_id=None):
            return self.add_trigger(img)

        def add_trigger(self, img):

            img_tensor = torch.tensor((np.array(img) / 255).transpose(2,0,1)).float()[None,...]

            img_tensor = disguise_embeddings(
                source_class_img_tensor = self.target_instance_tensor,
                target_class_img_tensor = img_tensor,
                net = self.net,
                device = self.device,
                max_iter = self.max_iter,
                terminal_loss = self.terminal_loss,
                lr = self.lr,
                beta = self.beta,
                final_blended_rate = self.final_blended_rate,
            )

            img = (img_tensor[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

            return img

    adv_train_ds = prepro_cls_DatasetBD( #TODO this implement is correct but slow need to speed up
        deepcopy(train_dataset_without_transform),
        poison_idx= train_pidx,
        bd_image_pre_transform=poisonFrogs(
            benign_train_dl.dataset.dataset[args.target_instance_index][0], #TODO
            net = net,
            device = device,
            max_iter = args.generate_max_iter,
            terminal_loss = args.generate_terminal_loss,
            lr = args.generate_lr,
            beta = args.generate_beta,
            final_blended_rate = args.generate_final_blended_rate,
        ),
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
    )

    adv_train_dl = DataLoader(
        dataset = adv_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )



    adv_test_dataset = TensorDataset(benign_train_dl.dataset[args.target_instance_index][0][None,...], torch.tensor(benign_train_dl.dataset[args.target_instance_index][1])[None,...])

    adv_test_dl = DataLoader(
        dataset = adv_test_dataset,
        batch_size= args.batch_size,
        shuffle= False,
        drop_last= False,
    )

    trainer = generate_cls_trainer(
        net,
        args.attack
    )



    criterion = argparser_criterion(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)

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