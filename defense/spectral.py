import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn

# TODO:怎么查看包的相对路径和绝对路径
sys.path.append('../')
sys.path.append(os.getcwd())

# TODO:修改yaml文件

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result

class spectral(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        # TODO:直接用self.args好不好用
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result_file(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/spectral/config.yaml", help='the path of yaml')

        #set the parameter for the spectral defense
        parser.add_argument('--percentile', type=float)
        parser.add_argument('--target_label', type=int)
        

    def set_result(self, result_file):
        save_path = './record/' + result_file
        assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + '/defense/spectral/checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + '/defense/spectral/log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(save_path + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = ModelTrainerCLS(
            model = model,
            amp = self.args.amp,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
   
    def mitigation(self):
        fix_random(self.args.random_seed)

        ### a. prepare the model and dataset
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        model.to(self.args.device)

        # Setting up the data and the model
        tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        x = self.result['bd_train']['x']
        y = self.result['bd_train']['y']
        data_set = list(zip(x,y))
        data_set_o = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_set,
            poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        dataset = data_set_o
    
        # initialize data augmentation
        logging.info(f'Dataset Size: {len(dataset)}' )

        if 'target_label' in args.__dict__:
            if isinstance(self.args.target_label,(int)):
                poison_labels = [self.args.target_label]
            else:
                poison_labels = self.args.target_label
        else:
            poison_labels = range(self.args.num_classes)

        re_all = []
        for target_label in poison_labels:
            lbl = target_label
            dataset_y = []
            for i in range(len(dataset)):
                dataset_y.append(dataset[i][1])
            cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
            cur_examples = len(cur_indices)
            logging.info(f'Label, num ex: {lbl},{cur_examples}' )
            
            model.eval()
            ### b. get the activation as representation for each data
            for iex in range(cur_examples):
                cur_im = cur_indices[iex]
                x_batch = dataset[cur_im][0].unsqueeze(0).to(self.args.device)
                y_batch = dataset[cur_im][1]
                with torch.no_grad():
                    assert self.args.model in ['preactresnet18', 'vgg19', 'resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3']
                    if self.args.model == 'preactresnet18':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.layer4.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                    elif self.args.model == 'vgg19':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.features.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                    elif self.args.model == 'resnet18':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.layer4.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                    elif self.args.model == 'mobilenet_v3_large':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.avgpool.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                    elif self.args.model == 'densenet161':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.features.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        outs[0] = torch.nn.functional.relu(outs[0])
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                    elif self.args.model == 'efficientnet_b3':
                        inps,outs = [],[]
                        def layer_hook(module, inp, out):
                            outs.append(out.data)
                        hook = model.avgpool.register_forward_hook(layer_hook)
                        _ = model(x_batch)
                        batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                        hook.remove()
                
                if iex==0:
                    full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
                full_cov[iex] = batch_grads.detach().cpu().numpy()

            ### c. detect the backdoor data by the SVD decomposition
            total_p = self.args.percentile            
            full_mean = np.mean(full_cov, axis=0, keepdims=True)            
        
            centered_cov = full_cov - full_mean
            u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
            logging.info(f'Top 7 Singular Values: {s[0:7]}')
            eigs = v[0:1]  
            p = total_p
            corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
            scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
            logging.info(f'Length Scores: {len(scores)}' )
            p_score = np.percentile(scores, p)
            top_scores = np.where(scores>p_score)[0]
            logging.info(f'{top_scores}')
            

            removed_inds = np.copy(top_scores)
            re = [cur_indices[v] for i,v in enumerate(removed_inds)]
            re_all.extend(re)
            
        left_inds = np.delete(range(len(dataset)), re_all)
        ### d. retrain the model with remaining data
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.to(self.args.device)
        dataset.subset(left_inds)
        dataset_left = dataset
        data_loader_sie = torch.utils.data.DataLoader(dataset_left, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        x = self.result['bd_test']['x']
        y = self.result['bd_test']['y']
        data_bd_test = list(zip(x,y))
        data_bd_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_bd_test,
            poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        x = self.result['clean_test']['x']
        y = self.result['clean_test']['y']
        data_clean_test = list(zip(x,y))
        data_clean_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_clean_test,
            poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        self.trainer.train_with_test_each_epoch(
            train_data = data_loader_sie,
            test_data = data_clean_loader,
            adv_test_data = data_bd_loader,
            end_epoch_num = self.args.epochs,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            device = self.args.device,
            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.checkpoint_save,
            save_prefix = 'defense',
            continue_training_path = None,
        )

        result = {}
        result["dataset"] = dataset_left
        result['model'] = model
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    spectral.add_arguments(parser)
    args = parser.parse_args()
    spectral_method = spectral(args)
    args.result_file = 'test_defense_badnet_attack_1epoch'
    result = spectral_method.defense(args.result_file)