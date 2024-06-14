# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''

Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering

This file is modified based on the following source:
link : https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py.
The defense method is called ac.

 @article{chen2018detecting,
        title={Detecting backdoor attacks on deep neural networks by activation clustering},
        author={Chen, Bryant and Carvalho, Wilka and Baracaldo, Nathalie and Ludwig, Heiko and Edwards, Benjamin and Lee, Taesung and Molloy, Ian and Srivastava, Biplav},
        journal={arXiv preprint arXiv:1811.03728},
        year={2018}
        }

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. during training the backdoor attack generalization to lower poison ratio (generalize_to_lower_pratio)
    5. save process
    6. new standard: robust accuracy
    7. reintegrate the framework
    8. hook the activation of the neural network
    9. add some addtional backbone such as preactresnet18, resnet18 and vgg19
    10. for data sets with many analogies, the classification bug existing in the original method is fixed
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ac defense:
        a. classify data by activation results
        b. identify backdoor data according to classification results
        c. retrain the model with filtered data
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.nCHW_nHWC import *

from sklearn.cluster import KMeans



class ac(defense):
    r"""Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. ac defense:
        a. classify data by activation results
        b. identify backdoor data according to classification results
        c. retrain the model with filtered data
    4. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        ac.add_arguments(parser)
        args = parser.parse_args()
        ac_method = ac(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = ac_method.defense(args.result_file)
    
    .. Note::
        @article{chen2018detecting,
        title={Detecting backdoor attacks on deep neural networks by activation clustering},
        author={Chen, Bryant and Carvalho, Wilka and Baracaldo, Nathalie and Ludwig, Heiko and Edwards, Benjamin and Lee, Taesung and Molloy, Ian and Srivastava, Biplav},
        journal={arXiv preprint arXiv:1811.03728},
        year={2018}
        }

    Args:
        baisc args: in the base class
        nb_dims (int): number of dimensions to reduce activation to by PCA.
        nb_clusters (int): number of clusters (defaults to 2 for poison/clean).
        cluster_analysis (str): the method of cluster analysis (smaller, relative-size, distance, silhouette-scores)
        cluster_batch_size (int): the batch size of cluster analysis
        
    """ 

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
        
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
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
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ac/config.yaml", help='the path of yaml')

        #set the parameter for the ac defense
        parser.add_argument('--nb_dims', type=int, help='number of dimensions to reduce activation to')
        parser.add_argument('--nb_clusters', type=int, help='number of clusters (defaults to 2 for poison/clean).')
        parser.add_argument('--cluster_analysis', type=str, help='the method of cluster analysis')
        parser.add_argument('--cluster_batch_size', type=int)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/ac/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
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
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        ### a. classify data by activation results
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        
        
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        train_dataset = self.result['bd_train'].wrapped_dataset
        data_set_without_tran = train_dataset
        data_set_o = self.result['bd_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_set_o.wrapped_dataset.getitem_all = False
        if not 'cluster_batch_size' in self.args.__dict__:
            self.args.cluster_batch_size = self.args.batch_size
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.cluster_batch_size, num_workers=self.args.num_workers, shuffle=True)
        num_classes = self.args.num_classes
        for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)
            batch_activations = get_activations(self.result['model_name'],model,x_batch.to(self.args.device))
            activation_dim = batch_activations.shape[-1]

            # initialize values list of lists on first run
            if i == 0:
                activations_by_class = [np.empty((0, activation_dim)) for _ in range(num_classes)]
                clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
                red_activations_by_class = [np.empty((0, self.args.nb_dims)) for _ in range(num_classes)]

            activations_by_class_i = segment_by_class(batch_activations, y_batch,self.args.num_classes)
            clusters_by_class_i, red_activations_by_class_i = cluster_activations(
                activations_by_class_i,
                nb_clusters=self.args.nb_clusters,
                nb_dims=self.args.nb_dims,
                reduce='PCA',
                clustering_method='KMeans'
            )

            for class_idx in range(num_classes):
                if activations_by_class_i[class_idx].shape[0] != 0:
                    activations_by_class[class_idx] = np.vstack(
                        [activations_by_class[class_idx], activations_by_class_i[class_idx]]
                    )
                    clusters_by_class[class_idx] = np.append(
                        clusters_by_class[class_idx], [clusters_by_class_i[class_idx]]
                    )
                    red_activations_by_class[class_idx] = np.vstack(
                        [red_activations_by_class[class_idx], red_activations_by_class_i[class_idx]]
                    )

        ### b. identify backdoor data according to classification results
        analyzer = ClusteringAnalyzer()
        if self.args.cluster_analysis == "smaller":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_size(clusters_by_class)
        elif self.args.cluster_analysis == "relative-size":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_relative_size(clusters_by_class)
        elif self.args.cluster_analysis == "distance":
            (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_distance(
                clusters_by_class,
                separated_activations=red_activations_by_class,
            )
        elif self.args.cluster_analysis == "silhouette-scores":
            (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_silhouette_score(
                clusters_by_class,
                reduced_activations_by_class=red_activations_by_class,
            )
        else:
            raise ValueError("Unsupported cluster analysis technique " + self.args.cluster_analysis)

        batch_size = self.args.cluster_batch_size
        is_clean_lst = []
        # loop though the generator to generator a report
        last_loc = torch.zeros(self.args.num_classes).numpy().astype(int)
        for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore
            indices_by_class = segment_by_class(np.arange(batch_size), y_batch,self.args.num_classes)
            is_clean_lst_i = [0] * batch_size
            clean_class = [0] * batch_size
            for class_idx, idxs in enumerate(indices_by_class):
                for idx_in_class, idx in enumerate(idxs):
                    is_clean_lst_i[idx] = assigned_clean_by_class[class_idx][idx_in_class + last_loc[class_idx]]
                last_loc[class_idx] = last_loc[class_idx] + len(idxs)
            is_clean_lst += is_clean_lst_i
        

        ### c. retrain the model with filtered data
        model = generate_cls_model(self.args.model,self.args.num_classes)
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        data_set_o.subset([i for i,v in enumerate(is_clean_lst) if v==1])
        data_set_o.wrapped_dataset.getitem_all = True
        data_loader_sie = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, drop_last=True)
        
        # optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        # self.set_trainer(model)
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(args)

        # test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        # x = self.result['bd_test']['x']
        # y = self.result['bd_test']['y']
        # data_bd_test = list(zip(x,y))
        # data_bd_testset = prepro_cls_DatasetBD(
        #     full_dataset_without_transform=data_bd_test,
        #     poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        #     bd_image_pre_transform=None,
        #     bd_label_pre_transform=None,
        #     ori_image_transform_in_loading=test_tran,
        #     ori_label_transform_in_loading=None,
        #     add_details_in_preprocess=False,
        # )
        # data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        # x = self.result['clean_test']['x']
        # y = self.result['clean_test']['y']
        # data_clean_test = list(zip(x,y))
        # data_clean_testset = prepro_cls_DatasetBD(
        #     full_dataset_without_transform=data_clean_test,
        #     poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        #     bd_image_pre_transform=None,
        #     bd_label_pre_transform=None,
        #     ori_image_transform_in_loading=test_tran,
        #     ori_label_transform_in_loading=None,
        #     add_details_in_preprocess=False,
        # )
        # data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)


        self.trainer.train_with_test_each_epoch_on_mix(
            data_loader_sie,
            data_clean_loader,
            data_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='ac',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )
        
        # self.trainer.train_with_test_each_epoch(
        #     train_data = data_loader_sie,
        #     test_data = data_clean_loader,
        #     adv_test_data = data_bd_loader,
        #     end_epoch_num = self.args.epochs,
        #     criterion = criterion,
        #     optimizer = optimizer,
        #     scheduler = scheduler,
        #     device = self.args.device,
        #     frequency_save = self.args.frequency_save,
        #     save_folder_path = self.args.checkpoint_save,
        #     save_prefix = 'defense',
        #     continue_training_path = None,
        # )

        # model.to(self.args.device)
        result = {}
        result['model'] = model
        result['dataset'] = data_set_o
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result     

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

def segment_by_class(data , classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    try:
        width = data.size()[1]
        by_class: List[List[int]] = [[] for _ in range(num_classes)]

        for indx, feature in enumerate(classes):
            if len(classes.shape) == 2 and classes.shape[1] > 1:

                assigned = np.argmax(feature)

            else:

                assigned = int(feature)
            if torch.is_tensor(data[indx]):
                by_class[assigned].append(data[indx].cpu().numpy())
            else:
                by_class[assigned].append(data[indx])
        return [np.asarray(i).reshape(-1,width) for i in by_class]
    except :
        by_class: List[List[int]] = [[] for _ in range(num_classes)]

        for indx, feature in enumerate(classes):
            if len(classes.shape) == 2 and classes.shape[1] > 1:

                assigned = np.argmax(feature)

            else:

                assigned = int(feature)
            if torch.is_tensor(data[indx]):
                by_class[assigned].append(data[indx].cpu().numpy())
            else:
                by_class[assigned].append(data[indx])
        return [np.asarray(i) for i in by_class]

def measure_misclassification(
    classifier, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Computes 1-accuracy given x_test and y_test
    :param classifier: Classifier to be used for predictions.
    :param x_test: Test set.
    :param y_test: Labels for test set.
    :return: 1-accuracy.
    """
    predictions = np.argmax(classifier.predict(x_test), axis=1)
    return 1.0 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

def train_remove_backdoor(
    classifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tolerable_backdoor: float,
    max_epochs: int,
    batch_epochs: int,
) -> tuple:
    """
    Trains the provider classifier until the tolerance or number of maximum epochs are reached.
    :param classifier: Classifier to be used for predictions.
    :param x_train: Training set.
    :param y_train: Labels used for training.
    :param x_test: Samples in test set.
    :param y_test: Labels in test set.
    :param tolerable_backdoor: Parameter that determines how many misclassifications are acceptable.
    :param max_epochs: maximum number of epochs to be run.
    :param batch_epochs: groups of epochs that will be run together before checking for termination.
    :return: (improve_factor, classifier).
    """
    # Measure poison success in current model:
    initial_missed = measure_misclassification(classifier, x_test, y_test)

    curr_epochs = 0
    curr_missed = 1.0
    while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
        classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
        curr_epochs += batch_epochs
        curr_missed = measure_misclassification(classifier, x_test, y_test)

    improve_factor = initial_missed - curr_missed
    return improve_factor, classifier


def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator = None,
    clusterer_new = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.
    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for activation in separated_activations:
        # Apply dimensionality reduction
        try :
            nb_activations = np.shape(activation)[1]
        except IndexError:
            activation = activation.reshape(1,-1)
            nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims & np.shape(activation)[0] > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        elif nb_activations <= nb_dims:
            reduced_activations = activation
        else:
            reduced_activations = activation[:,0:(nb_dims)]
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None and reduced_activations.shape[0] != 0:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        elif reduced_activations.shape[0] != 1 and reduced_activations.shape[0] != 0:
            clusters = clusterer.fit_predict(reduced_activations)
        else:
            clusters = 1
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations


def reduce_dimensionality(activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA") -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.
    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA

    if reduce == "FastICA":
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations

def get_activations(name,model,x_batch):
    ''' get activations of the model for each sample
    name:
        the model name 
    model:
        the train model
    x_batch:
        each batch for tain data
    '''
    with torch.no_grad():
        model.eval()
        TOO_SMALL_ACTIVATIONS = 32
        assert name in ['preactresnet18', 'vgg19','vgg19_bn', 'resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3','convnext_tiny','vit_b_16']
        if name == 'preactresnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'vgg19':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'vgg19_bn':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'resnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'mobilenet_v3_large':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'densenet161':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            outs[0] = torch.nn.functional.relu(outs[0])
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'efficientnet_b3':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'convnext_tiny':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == 'vit_b_16':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                inps.append(inp[0].data)
            hook = model[1].heads.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = inps[0].view(inps[0].size(0), -1)
            hook.remove()


    return activations


# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements methodologies to analyze clusters and determine whether they are poisonous.
"""

class ClusteringAnalyzer:

    """
    Class for all methodologies implemented to analyze clusters and determine whether they are poisonous.
    """
    @staticmethod
    def assign_class(clusters: np.ndarray, clean_clusters: List[int], poison_clusters: List[int]) -> np.ndarray:

        """
        Determines whether each data point in the class is in a clean or poisonous cluster
        :param clusters: `clusters[i]` indicates which cluster the i'th data point is in.
        :param clean_clusters: List containing the clusters designated as clean.
        :param poison_clusters: List containing the clusters designated as poisonous.
        :return: assigned_clean: `assigned_clean[i]` is a boolean indicating whether the ith data point is clean.
        """

        assigned_clean = np.empty(np.shape(clusters))
        assigned_clean[np.isin(clusters, clean_clusters)] = 1
        assigned_clean[np.isin(clusters, poison_clusters)] = 0
        return assigned_clean



    def analyze_by_size(
        self, separated_clusters: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:

        """
        Designates as poisonous the cluster with less number of items on it.
        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether
                 a given data point was determined to be clean (as opposed to poisonous) and
                 summary_poison_clusters: array, where summary_poison_clusters[i][j]=1 if cluster j of class i was
                 classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis
        """
        report: Dict[str, Any] = {
            "cluster_analysis": "smaller",
            "suspicious_clusters": 0,
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [[0 for _ in range(nb_clusters)] for _ in range(nb_classes)]

        for i, clusters in enumerate(separated_clusters):
            # assume that smallest cluster is poisonous and all others are clean
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)
            poison_clusters: List[int] = [int(np.argmin(sizes))]
            clean_clusters = list(set(clusters) - set(poison_clusters))
            for p_id in poison_clusters:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters:
                summary_poison_clusters[i][c_id] = 0



            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)
            # Generate report for this class:
            report_class = dict()
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp)
                dict_cluster: Dict[str, Dict[str, int]] = {"cluster_" + str(cluster_id): dict_i}
                report_class.update(dict_cluster)

            report["Class_" + str(i)] = report_class

        report["suspicious_clusters"] = report["suspicious_clusters"] + np.sum(summary_poison_clusters).item()
        return np.asarray(all_assigned_clean), summary_poison_clusters, report

    def analyze_by_distance(
        self,
        separated_clusters: List[np.ndarray],
        separated_activations: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:

        """
        Assigns a cluster as poisonous if its median activation is closer to the median activation for another class
        than it is to the median activation of its own class. Currently, this function assumes there are only two
        clusters per class.
        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.
        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean].
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis.
        """

        report: Dict[str, Any] = {"cluster_analysis": 0.0}
        all_assigned_clean = []
        cluster_centers = []

        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [[0 for _ in range(nb_clusters)] for _ in range(nb_classes)]

        # assign centers
        for _, activations in enumerate(separated_activations):
            cluster_centers.append(np.median(activations, axis=0))

        for i, (clusters, activation) in enumerate(zip(separated_clusters, separated_activations)):
            clusters = np.array(clusters)
            cluster0_center = np.median(activation[np.where(clusters == 0)], axis=0)
            cluster1_center = np.median(activation[np.where(clusters == 1)], axis=0)

            cluster0_distance = np.linalg.norm(cluster0_center - cluster_centers[i])
            cluster1_distance = np.linalg.norm(cluster1_center - cluster_centers[i])

            cluster0_is_poison = False
            cluster1_is_poison = False

            dict_k = dict()
            dict_cluster_0 = dict(cluster0_distance_to_its_class=str(cluster0_distance))
            dict_cluster_1 = dict(cluster1_distance_to_its_class=str(cluster1_distance))
            for k, center in enumerate(cluster_centers):
                if k == i:
                    pass
                else:
                    cluster0_distance_to_k = np.linalg.norm(cluster0_center - center)
                    cluster1_distance_to_k = np.linalg.norm(cluster1_center - center)
                    if cluster0_distance_to_k < cluster0_distance and cluster1_distance_to_k > cluster1_distance:
                        cluster0_is_poison = True
                    if cluster1_distance_to_k < cluster1_distance and cluster0_distance_to_k > cluster0_distance:
                        cluster1_is_poison = True

                    dict_cluster_0["distance_to_class_" + str(k)] = str(cluster0_distance_to_k)
                    dict_cluster_0["suspicious"] = str(cluster0_is_poison)
                    dict_cluster_1["distance_to_class_" + str(k)] = str(cluster1_distance_to_k)
                    dict_cluster_1["suspicious"] = str(cluster1_is_poison)
                    dict_k.update(dict_cluster_0)
                    dict_k.update(dict_cluster_1)



            report_class = dict(cluster_0=dict_cluster_0, cluster_1=dict_cluster_1)
            report["Class_" + str(i)] = report_class

            poison_clusters = []
            if cluster0_is_poison:
                poison_clusters.append(0)
                summary_poison_clusters[i][0] = 1
            else:
                summary_poison_clusters[i][0] = 0

            if cluster1_is_poison:
                poison_clusters.append(1)
                summary_poison_clusters[i][1] = 1
            else:
                summary_poison_clusters[i][1] = 0

            clean_clusters = list(set(clusters) - set(poison_clusters))
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        all_assigned_clean = np.asarray(all_assigned_clean)
        return all_assigned_clean, summary_poison_clusters, report

    def analyze_by_relative_size(
        self,
        separated_clusters: List[np.ndarray],
        size_threshold: float = 0.35,
        r_size: int = 2,
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:

        """
        Assigns a cluster as poisonous if the smaller one contains less than threshold of the data.
        This method assumes only 2 clusters
        :param separated_clusters: List where `separated_clusters[i]` is the cluster assignments for the ith class.
        :param size_threshold: Threshold used to define when a cluster is substantially smaller.
        :param r_size: Round number used for size rate comparisons.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis.
        """

        size_threshold = round(size_threshold, r_size)
        report: Dict[str, Any] = {
            "cluster_analysis": "relative_size",
            "suspicious_clusters": 0,
            "size_threshold": size_threshold,
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [[0 for _ in range(nb_clusters)] for _ in range(nb_classes)]

        for i, clusters in enumerate(separated_clusters):
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)

            if np.size(sizes) > 2:
                raise ValueError(" RelativeSizeAnalyzer does not support more than two clusters.")
            percentages = np.round(sizes / float(np.sum(sizes)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0



            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

            # Generate report for this class:
            report_class = dict()
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp)

                dict_cluster = {"cluster_" + str(cluster_id): dict_i}
                report_class.update(dict_cluster)

            report["Class_" + str(i)] = report_class

        report["suspicious_clusters"] = report["suspicious_clusters"] + np.sum(summary_poison_clusters).item()
        return np.asarray(all_assigned_clean), summary_poison_clusters, report

    def analyze_by_silhouette_score(
        self,
        separated_clusters: list,
        reduced_activations_by_class: list,
        size_threshold: float = 0.35,
        silhouette_threshold: float = 0.1,
        r_size: int = 2,
        r_silhouette: int = 4,
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:

        """
        Analyzes clusters to determine level of suspiciousness of poison based on the cluster's relative size
        and silhouette score.
        Computes a silhouette score for each class to determine how cohesive resulting clusters are.
        A low silhouette score indicates that the clustering does not fit the data well, and the class can be considered
        to be un-poisoned. Conversely, a high silhouette score indicates that the clusters reflect true splits in the
        data.
        The method concludes that a cluster is poison based on the silhouette score and the cluster relative size.
        If the relative size is too small, below a size_threshold and at the same time
        the silhouette score is higher than silhouette_threshold, the cluster is classified as poisonous.
        If the above thresholds are not provided, the default ones will be used.
        :param separated_clusters: list where `separated_clusters[i]` is the cluster assignments for the ith class.
        :param reduced_activations_by_class: list where separated_activations[i] is a 1D array of [0,1] for
               [poison,clean].
        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :param silhouette_threshold: (optional) threshold used to define when a cluster is cohesive. Default
        value is used if the parameter is not provided.
        :param r_size: Round number used for size rate comparisons.
        :param r_silhouette: Round number used for silhouette rate comparisons.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class j was classified as poison
                 report: Dictionary with summary of the analysis.
        """

        # pylint: disable=E0001
        from sklearn.metrics import silhouette_score
        size_threshold = round(size_threshold, r_size)
        silhouette_threshold = round(silhouette_threshold, r_silhouette)
        report: Dict[str, Any] = {
            "cluster_analysis": "silhouette_score",
            "size_threshold": str(size_threshold),
            "silhouette_threshold": str(silhouette_threshold),
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [[0 for _ in range(nb_clusters)] for _ in range(nb_classes)]

        for i, (clusters, activations) in enumerate(zip(separated_clusters, reduced_activations_by_class)):

            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError("Analyzer does not support more than two clusters.")

            percentages = np.round(bins / float(np.sum(bins)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            # Generate report for class
            silhouette_avg = round(silhouette_score(activations, clusters), r_silhouette)
            dict_i: Dict[str, Any] = dict(
                sizes_clusters=str(bins),
                ptc_cluster=str(percentages),
                avg_silhouette_score=str(silhouette_avg),
            )

            if np.shape(poison_clusters)[1] != 0:
                # Relative size of the clusters is suspicious
                if silhouette_avg > silhouette_threshold:
                    # In this case the cluster is considered poisonous
                    clean_clusters = np.where(percentages < size_threshold)
                    logging.info("computed silhouette score: %s", silhouette_avg)
                    dict_i.update(suspicious=True)
                else:
                    poison_clusters = [[]]
                    clean_clusters = np.where(percentages >= 0)
                    dict_i.update(suspicious=False)
            else:
                # If relative size of the clusters is Not suspicious, we conclude it's not suspicious.

                dict_i.update(suspicious=False)

            report_class: Dict[str, Dict[str, bool]] = {"class_" + str(i): dict_i}
            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1

            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)
            report.update(report_class)

        return np.asarray(all_assigned_clean), summary_poison_clusters, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ac.add_arguments(parser)
    args = parser.parse_args()
    ac_method = ac(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ac_method.defense(args.result_file)