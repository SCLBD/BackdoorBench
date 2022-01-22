




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
This module implements methods performing poisoning detection based on activations clustering.
| Paper link: https://arxiv.org/abs/1811.03728
| Please keep in mind the limitations of defences. For more information on the limitations of this
    defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
    in general, see https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import logging
import argparse
import sys
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from clustering_analyzer import ClusteringAnalyzer

sys.path.append(os.getcwd())
from utils.dataloader_bd import get_dataloader_train,get_dataloader_test,get_dataset_train,DatasetStd
from utils.network import get_network
import yaml


def segment_by_class(data: Union[np.ndarray, List[int]], classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    by_class: List[List[int]] = [[] for _ in range(num_classes)]

    for indx, feature in enumerate(classes):
        if len(classes.shape) == 2 and classes.shape[1] > 1:

            assigned = np.argmax(feature)

        else:

            assigned = int(feature)

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
        nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        else:
            reduced_activations = activation
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        else:
            clusters = clusterer.fit_predict(reduced_activations)
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


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--checkpoint_save', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument("--data_root", type=str, default='dataset/')

    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--attack', type=str, default='badnet')
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='squareTrigger', help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')


    #####AC
    parser.add_argument('--nb_dims', type=int, default=None, help='train epoch')
    parser.add_argument('--nb_clusters', type=int, default=None, help='the number of mini_batch train model')
    parser.add_argument('--cluster_analysis', type=str, default=None, help='the method of cluster analysis')
    



    arg = parser.parse_args()

    if arg.dataset == "mnist":
        arg.num_classes = 10
        arg.input_height = 28
        arg.input_width = 28
        arg.input_channel = 1
    elif arg.dataset == "cifar10":
        arg.num_classes = 10
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "gtsrb":
        arg.num_classes = 43
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "celeba":
        arg.num_classes = 8
        arg.input_height = 64
        arg.input_width = 64
        arg.input_channel = 3
    elif arg.dataset == "tiny":
        arg.num_classes = 200
        arg.input_height = 64
        arg.input_width = 64
        arg.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    arg.checkpoint_save = 'saved/checkpoint/checkpoint_' + arg.dataset + '.tar'
    arg.log = 'saved/log/log_' + arg.dataset + '.txt'
    arg.data_root = arg.data_root + arg.dataset    
    if not os.path.isdir(arg.data_root):
        os.makedirs(arg.data_root)
    print(arg)
    return arg



def ac(args,model):
    with open("./config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 

    ####写个替换的
    nb_dims = config['nb_dims']
    nb_clusters = config['nb_clusters']
    cluster_analysis = config['cluster_analysis']
    ###cluster
   
    data_set = get_dataset_train(args)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    batch_size = args.batch_size
    num_samples = len(data_loader.dataset).size
    num_classes = args.num_classes
    for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore

        batch_activations = get_activations(model,x_batch.to(args.device))
        activation_dim = batch_activations.shape[-1]

        # initialize values list of lists on first run
        if i == 0:
            activations_by_class = [np.empty((0, activation_dim)) for _ in range(num_classes)]
            clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
            red_activations_by_class = [np.empty((0, nb_dims)) for _ in range(num_classes)]

        activations_by_class_i = segment_by_class(batch_activations, y_batch,args.num_class)
        clusters_by_class_i, red_activations_by_class_i = cluster_activations(
            activations_by_class_i,
            nb_clusters=nb_clusters,
            nb_dims=nb_dims,
            reduce='PCA',
            clustering_method='KMeans'
        )

        for class_idx in range(num_classes):
            activations_by_class[class_idx] = np.vstack(
                [activations_by_class[class_idx], activations_by_class_i[class_idx]]
            )
            clusters_by_class[class_idx] = np.append(
                [clusters_by_class[class_idx], clusters_by_class_i[class_idx]]
            )
            red_activations_by_class[class_idx] = np.vstack(
                [red_activations_by_class[class_idx], red_activations_by_class_i[class_idx]]
            )

    ###analyze
    analyzer = ClusteringAnalyzer()
    if cluster_analysis == "smaller":
        (
            assigned_clean_by_class,
            poisonous_clusters,
            report,
        ) = analyzer.analyze_by_size(clusters_by_class)
    elif cluster_analysis == "relative-size":
        (
            assigned_clean_by_class,
            poisonous_clusters,
            report,
        ) = analyzer.analyze_by_relative_size(clusters_by_class)
    elif cluster_analysis == "distance":
        (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_distance(
            clusters_by_class,
            separated_activations=red_activations_by_class,
        )
    elif cluster_analysis == "silhouette-scores":
        (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_silhouette_score(
            clusters_by_class,
            reduced_activations_by_class=red_activations_by_class,
        )
    else:
        raise ValueError("Unsupported cluster analysis technique " + cluster_analysis)

    ###detect

    batch_size = args.batch_size
    is_clean_lst = []

    # loop though the generator to generator a report
    for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore
        indices_by_class = segment_by_class(np.arange(batch_size), y_batch,args.num_classes)
        is_clean_lst_i = [0] * batch_size
        for class_idx, idxs in enumerate(indices_by_class):
            for idx_in_class, idx in enumerate(idxs):
                is_clean_lst_i[idx] = assigned_clean_by_class[class_idx][idx_in_class]
        is_clean_lst += is_clean_lst_i
    

    ###reliable
    dataset_sie_ = []
    for i in range(num_samples):
        if is_clean_lst[i] == 1:
            dataset_sie_.append[data_set[i]]
        else:
            dataset_sie_.append[data_set[i][0],class_idx[i]]
    dataset_sie = DatasetStd(dataset_sie_)
    data_loader_sie = torch.utils.data.DataLoader(dataset_sie, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    criterion = nn.CrossEntropyLoss() 
    for j in range(args.epochs):
        for i, (inputs,labels) in enumerate(data_loader_sie):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    result['model'] = model
    return result       

   

def get_activations(model,x_batch):
    TOO_SMALL_ACTIVATIONS = 32
    if model.layer_names is not None:
        nb_layers = len(model.layer_names)
    else:
        raise ValueError("No layer names identified.")
    protected_layer = nb_layers - 1

    activations = model.get_activations(x_batch, layer=protected_layer)

    nodes_last_layer = np.shape(activations)[1]

    if nodes_last_layer <= TOO_SMALL_ACTIVATIONS:
        logger.warning(
            "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
            str(nodes_last_layer),
        )
    return activations




if __name__ == '__main__':
    
    args = get_args()
    model = get_network(args)
    if args.load_target is not None:
        checkpoint = torch.load(args.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        result = ac(args,model,args.method)
    else:
        print("There is no target model")