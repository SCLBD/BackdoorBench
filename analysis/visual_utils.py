import os
import torch
import copy
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
    dataset_and_transform_generate
)
import contextlib


def get_args(use_IPython=False):
    # set the basic parameter
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="cuda|cpu")
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./config/visualization/default.yaml",
        help="the path of yaml which contains the default parameters",
    )
    parser.add_argument("--seed", type=str, help="random seed for reproducibility")
    parser.add_argument("--model", type=str, help="model name such as resnet18, vgg19")

    # data parameters
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    parser.add_argument(
        "--dataset", type=str, help="mnist, cifar10, cifar100, gtsrb, celeba, tiny"
    )
    parser.add_argument("--visual_dataset", type=str, default='bd_train',
                        help="type of dataset for visualization. mixed|clean_train|clean_test|bd_train|bd_test")
    parser.add_argument("--target_class", type=int,
                        default=0, help="tagrt class for attack, used for subset dataset, legend, title, etc.")
    parser.add_argument("--num_classes", type=int, help="number of classes for given dataset used for create visualization dataset")
    parser.add_argument("--input_height", type=int, help="input height of the image")
    parser.add_argument("--input_width", type=int, help="input width of the image")
    parser.add_argument("--input_channel", type=int, help="input channel of the image")
    parser.add_argument("--batch_size", type=int, help="batch size for visualization")
    parser.add_argument("--n_sub", default=5000, type=int, help="max number of samples for visualization")
    parser.add_argument("--c_sub", default=10, type=int, help="max number of classes for visualization")
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers for dataloader")
    parser.add_argument("--class_names", type=list,
                        help="no need to give, it will be created by preprocess_args if not provided")

    # BD parameters
    parser.add_argument("--pratio", type=float, help="ratio of poisoned samples, used for mix_dataset and legend")

    # results parameters
    parser.add_argument(
        "--result_file_attack",
        default='badnet_demo',
        type=str,
        help="the location of attack result, must be given to load the dataset",
    )
    parser.add_argument(
        "--result_file_defense",
        default='None',
        type=str,
        help="the location of defense result. If given, the defense model will be used instead of the attack model",
    )
    parser.add_argument("--checkpoint_load", default=None, type=str)
    parser.add_argument("--checkpoint_save", default=None, type=str)

    # plot parameters
    parser.add_argument('--neuron_order', type=str, default='ordered',
                        help='The order of Neuron for visualization, used for visual_act.')
    parser.add_argument('--num_neuron', type=int, default=50,
                        help='The number of Neuron for visualization. Must less than 100, used for visual_act.')
    parser.add_argument('--num_image', type=int, default=10,
                        help='The number of images for each Neuron. Must less than 100, used for visual_act.')
    
    parser.add_argument('--target_layer_name', type=str, default='default',
                        help='The name of layer for extracting features, used by plots that use features.')

    # Parameter for Landscape Visualization
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--raw_data', action='store_true',
                        default=False, help='no data preprocessing')

    # direction parameters
    parser.add_argument('--dir_type', default='weights',
                        help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:30',
                        help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:30',
                        help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='filter',
                        help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='filter',
                        help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='biasbn',
                        help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='biasbn',
                        help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False,
                        help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int,
                        help='the index for the repeatness experiment')

    # plot parameters
    parser.add_argument('--loss_name', '-l', default='crossentropy',
                        help='loss functions: crossentropy | mse, used for landscape visualization')
    parser.add_argument('--metric_z', default='train_loss',
                        type=str, help='metric for z axis: train_loss | train_acc, used for landscape visualization')

    # Parameter for TAC and Lips plot
    parser.add_argument('--normalize_by_layer', action='store_true',
                        default=False, help='Normalize the values by layer, used for TAC and Lips plot')

    # For prototype
    parser.add_argument('--prototype', action='store_true',
                        default=False, help='Specify whether the result is for prototype')

    # whether use IPython like juptyer notebook
    if use_IPython:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


def preprocess_args(args):
    # preprocess args for dataset
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
        args.class_names = get_class_name(args.dataset, args.num_classes, args)

    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.class_names = get_class_name(args.dataset, args.num_classes, args)

    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.class_names = get_class_name(args.dataset, args.num_classes, args)

    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.class_names = get_class_name(args.dataset, args.num_classes, args)

    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
        args.class_names = get_class_name(args.dataset, args.num_classes, args)

    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
        args.class_names = get_class_name(args.dataset, args.num_classes, args)
    else:
        raise Exception("Invalid Dataset")

    # preprocess args for target layer
    if args.target_layer_name == 'default':
        if args.model == "preactresnet18":
            args.target_layer_name = 'layer4.1.conv2'
        if args.model == "vgg19":
            args.target_layer_name = 'features.34'
        if args.model == "resnet18":
            args.target_layer_name = 'layer4.1.conv2'
        if args.model == "densenet161":
            args.target_layer_name = 'features.denseblock4.denselayer24.conv2'
        if args.model == "mobilenet_v3_large":
            args.target_layer_name = 'features.16.0'
        if args.model == "efficientnet_b3":
            args.target_layer_name = 'features.7.1.block.3.0'

    # Preprofess args for landscape
    args.cuda = True if 'cuda' in args.device else False
    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.ymin, args.ymax, args.ynum = (None, None, None)
    args.xnum = int(args.xnum)
    if args.y:
        args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
        assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
        args.ynum = int(args.ynum)
    return args


@contextlib.contextmanager
def temporary_all_clean(bd_dataset):
    old_poison_indicator = bd_dataset.wrapped_dataset.poison_indicator.copy()
    bd_dataset.wrapped_dataset.poison_indicator = np.zeros_like(
        old_poison_indicator)
    try:
        yield bd_dataset
    finally:
        bd_dataset.wrapped_dataset.poison_indicator = old_poison_indicator


def get_true_label_from_bd_dataset(bd_dataset):
    # return the true label of a given BD dataset in the oder of item index
    # original idx, indicator, true label
    return [other_info[2] for img, label, *other_info in bd_dataset]


def get_poison_indicator_from_bd_dataset(bd_dataset):
    '''
    What's the difference between this function and dataset.poison_indicator?
    dataset.poison_indicator is always has the same length as the underlying full dataset and in the order of the full dataset
    this function returns the poison indicator in the order of the item order of bd_dataset, i.e., the index we get the samples from the bd_dataset
    Note: another way to get poison indicator is the get_feature function which returns the indicator in the order of (shuffled) dataloader
    '''
    # return the position of bd samples in the oder of item index
    # original idx, indicator, true label
    return [other_info[1] for img, label, *other_info in bd_dataset]

def get_index_mapping_from_bd_dataset(bd_dataset):
    '''
    A function to get the mapping from the index of the bd_dataset to the index of the full dataset.
    '''
    # return the position of bd samples in the oder of item index
    # original idx, indicator, true label
    ori_to_bd_idx = {}
    bd_idx_to_ori = {}
    for idx, (img, label, *other_info) in enumerate(bd_dataset):
        ori_to_bd_idx[other_info[0]] = idx
        bd_idx_to_ori[idx] = other_info[0]
        
    return ori_to_bd_idx, bd_idx_to_ori

def generate_clean_dataset(clean_dataset, selected_classes, max_num_samples):
    '''
    This function modifies clean datsaet to generate a new dataset with given selected classes and max number of samples.
    To do so, we change
    1. create a new dataset with clean samples in a BD dataset class by prepro_cls_DatasetBD_v2
    2. subset the dataset to the given max number of samples and classes
    '''
    # deepcopy the given dataset to avoid changing the original dataset
    clean_dataset_without_trans = copy.deepcopy(clean_dataset.wrapped_dataset)
    clean_dataset_without_trans = prepro_cls_DatasetBD_v2(
        clean_dataset_without_trans)
    assert np.sum(
        clean_dataset_without_trans.poison_indicator) == 0, "The given clean dataset is not clean."

    # subset the clean dataset to the given number of samples and classes
    true_labels = np.array(
        get_true_label_from_bd_dataset(clean_dataset_without_trans))
    subset_index = sub_sample_euqal_ratio_classes_index(
        true_labels, max_num_samples=max_num_samples, selected_classes=selected_classes)
    clean_dataset_without_trans.subset(subset_index)
    print('subset clean dataset with length: ',
          len(clean_dataset_without_trans))

    clean_dataset_with_trans = dataset_wrapper_with_transform(
        clean_dataset_without_trans,
        clean_dataset.wrap_img_transform,
        clean_dataset.wrap_label_transform,
    )
    return clean_dataset_with_trans


def generate_bd_dataset(bd_dataset, target_class, selected_classes, max_num_samples, bd_only=False):
    '''
    This function modifies BD datsaet to generate a new dataset with given selected classes and max number of samples.
    To do so, we change
    1. create a copy of BD dataset
    2. subset the dataset to the given max number of samples and classes
    '''
    # deepcopy the given dataset to avoid changing the original dataset
    bd_dataset_without_trans = copy.deepcopy(bd_dataset.wrapped_dataset)

    if bd_only:
        dataset_poi_indicator = np.array(
            get_poison_indicator_from_bd_dataset(bd_dataset_without_trans))
        bd_dataset_without_trans.subset(
            np.where(dataset_poi_indicator == 1)[0])

    true_bd_labels = np.array(
        get_true_label_from_bd_dataset(bd_dataset_without_trans))
    dataset_poi_indicator = get_poison_indicator_from_bd_dataset(
        bd_dataset_without_trans)
    lables_with_poi = true_bd_labels.copy()

    if not bd_only:
        # regard the poisoned samples as a new class -1
        lables_with_poi[dataset_poi_indicator == 1] = -1
        selected_classes = np.append(selected_classes, -1)

    # subset the mix dataset to the given number of samples and classes
    subset_index = sub_sample_euqal_ratio_classes_index(
        lables_with_poi, max_num_samples=max_num_samples, selected_classes=selected_classes)

    bd_dataset_without_trans.subset(subset_index)
    print('subset bd dataset with length: ', len(bd_dataset_without_trans))

    bd_dataset_with_trans = dataset_wrapper_with_transform(
        bd_dataset_without_trans,
        bd_dataset.wrap_img_transform,
        bd_dataset.wrap_label_transform,
    )
    return bd_dataset_with_trans


def generate_mix_dataset(bd_test, target_class, pratio, selected_classes, max_num_samples):
    '''
    This function modifies the bd_test_with_trans which has all poisoned non-target test samples by 
    1. changing the poison indictor to recover some clean samples/get_labels
    2. changing the original_index_array to recover the target samples 
    '''
    # deepcopy the given dataset to avoid changing the original dataset
    mix_dataset_without_trans = copy.deepcopy(bd_test.wrapped_dataset)
    mix_dataset_without_trans.original_index_array = np.arange(
        len(mix_dataset_without_trans.dataset))
    print('create mix dataset with length: ', len(mix_dataset_without_trans))

    # recover target classes
    true_bd_labels = np.array(
        get_true_label_from_bd_dataset(mix_dataset_without_trans))

    # random choose pratio of bd samples
    non_target_clean_idx = np.where(
        mix_dataset_without_trans.poison_indicator == 1)[0]
    bd_idx = np.random.choice(non_target_clean_idx, int(
        len(mix_dataset_without_trans)*pratio), replace=False)
    poi_indicators = np.zeros(len(mix_dataset_without_trans.dataset))
    poi_indicators[bd_idx] = 1

    # set selected poisoned samples
    mix_dataset_without_trans.poison_indicator = poi_indicators

    # regard the poisoned samples as a new class -1
    lables_with_poi = true_bd_labels.copy()
    lables_with_poi[bd_idx] = -1
    selected_classes = np.append(selected_classes, -1)

    # subset the mix dataset to the given number of samples and classes
    subset_index = sub_sample_euqal_ratio_classes_index(
        lables_with_poi, max_num_samples=max_num_samples, selected_classes=selected_classes)
    mix_dataset_without_trans.subset(subset_index)
    print('subset mix dataset with length: ', len(mix_dataset_without_trans))

    mix_test_dataset_with_trans = dataset_wrapper_with_transform(
        mix_dataset_without_trans,
        bd_test.wrap_img_transform,
        bd_test.wrap_label_transform,
    )
    return mix_test_dataset_with_trans


def load_prototype_result(args, save_path_attack):
    result_prototype = {}
    result_prototype['model_name'] = args.model
    result_prototype['num_classes'] = args.num_classes
    result_prototype['model'] = torch.load(save_path_attack + "/clean_model.pth")
    result_prototype['data_path'] =  args.dataset_path + "/" + args.dataset
    result_prototype['img_size'] = args.img_size = (args.input_height, args.input_width, args.input_channel)
    
    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args)

    clean_train_dataset_with_trans = dataset_wrapper_with_transform(
        train_dataset_without_transform,
        train_img_transform,
        train_label_transform,
    )

    clean_test_dataset_with_trans = dataset_wrapper_with_transform(
        test_dataset_without_transform,
        test_img_transform,
        test_label_transform,
    )
                
    result_prototype['clean_train'] = clean_train_dataset_with_trans
    result_prototype['clean_test'] = clean_test_dataset_with_trans
    
    # By default, we do not save bd_train and bd_test, you can change this setting if you need.
    result_prototype['bd_train'] = None
    result_prototype['bd_test'] = None
    
    return result_prototype    

def get_features(args, model, target_layer, data_loader, reduction='flatten', activation=None):
    '''Function to extract the features/embeddings/activations from a target layer'''

    # extract feature vector from a specific layer
    # output_ is of shape (num_samples, num_neurons, feature_map_width, feature_map_height), here we choose the max activation
    if reduction == 'flatten':
        def feature_hook(module, input_, output_):
            global feature_vector
            # access the layer output and convert it to a feature vector
            feature_vector = output_
            if activation is not None:
                feature_vector = activation(feature_vector)
            feature_vector = torch.flatten(feature_vector, 1)
            return None
    elif reduction == 'none':
        def feature_hook(module, input_, output_):
            global feature_vector
            # access the layer output and convert it to a feature vector
            feature_vector = output_
            if activation is not None:
                feature_vector = activation(feature_vector)
            feature_vector = feature_vector
            return None
    elif reduction == 'max':
        def feature_hook(module, input_, output_):
            global feature_vector
            # access the layer output and convert it to a feature vector
            feature_vector = output_
            if activation is not None:
                feature_vector = activation(feature_vector)
            if feature_vector.dim() > 2:
                feature_vector = torch.max(
                    torch.flatten(feature_vector, 2), 2)[0]
            else:
                feature_vector = feature_vector
            return None
    elif reduction == 'sum':
        def feature_hook(module, input_, output_):
            global feature_vector
            # access the layer output and convert it to a feature vector
            feature_vector = output_
            if activation is not None:
                feature_vector = activation(feature_vector)
            if feature_vector.dim() > 2:
                feature_vector = torch.sum(torch.flatten(feature_vector, 2), 2)
            else:
                feature_vector = feature_vector
            return None

    h = target_layer.register_forward_hook(feature_hook)

    model.eval()
    # collect feature vectors
    features = []
    labels = []
    poi_indicator = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, *other_info) in enumerate(data_loader):
            global feature_vector
            # Fetch features
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            # if activation is not None:
            #     feature_vector = activation(feature_vector)
            # move all tensor to cpu to save memory
            current_feature = feature_vector.detach().cpu().numpy()
            current_labels = targets.cpu().numpy()
            current_poi_indicator = np.array(other_info[1].numpy())

            # Store features
            features.append(current_feature)
            labels.append(current_labels)
            poi_indicator.append(current_poi_indicator)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    poi_indicator = np.concatenate(poi_indicator, axis=0)
    h.remove()  # Rmove the hook

    return features, labels, poi_indicator


def plot_embedding(
    tsne_result, label, title, xlabel="tsne_x", ylabel="tsne_y", custom_palette=None, size=(10, 10), mark_size = 40, alpha = 0.6
):
    """Plot embedding for T-SNE with labels"""
    # Data Preprocessing
    if torch.is_tensor(tsne_result):
        tsne_result = tsne_result.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    x_min, x_max = np.min(tsne_result, 0), np.max(tsne_result, 0)
    tsne_result = (tsne_result - x_min) / (x_max - x_min)

    # Plot
    tsne_result_df = pd.DataFrame(
        {"feature_x": tsne_result[:, 0],
            "feature_y": tsne_result[:, 1], "label": label}
    )
    fig, ax = plt.subplots(1, figsize=size)

    num_class = len(pd.unique(tsne_result_df["label"]))
    if custom_palette is None:
        custom_palette = sns.color_palette("hls", num_class)

    # s: maker size, palette: colors

    sns.scatterplot(
        x="feature_x",
        y="feature_y",
        hue="label",
        data=tsne_result_df,
        ax=ax,
        s=mark_size,
        palette=custom_palette,
        alpha=alpha,
    )
    #     sns.lmplot(x='feature_x', y='feature_y', hue='label',
    #                     data=tsne_result_df, size=9, scatter_kws={"s":20,"alpha":0.3},fit_reg=False, legend=True,)

    # Set Figure Style
    lim = (-0.01, 1.01)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.tick_params(axis="x", labelsize=20)
    #ax.tick_params(axis="y", labelsize=20)
    ax.set_title(title)
    ax.set_aspect("equal")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    return fig


def get_embedding(data, method = "tsne"):
    """Get T-SNE embeddings"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if method == "tsne":
        tsne = TSNE(n_components=2, init="random", random_state=0)
        result = tsne.fit_transform(data)
    elif method == "umap":
        umap = UMAP(n_components=2, init = 'spectral', random_state=0, metric = "euclidean")
        result = umap.fit_transform(data)
    else:
        assert False, "Illegal method"
    return result

def umap_fig(
    data,
    label,
    title="UMAP embedding",
    xlabel="umap_x",
    ylabel="umap_y",
    custom_palette=None,
    size=(10, 10),
    mark_size = 0.3,
    alpha = 0.8
):
    """Get UMAP embeddings figure"""
    umap_result = get_embedding(data, method = "umap")
    fig = plot_embedding(umap_result, label, title, xlabel,
                         ylabel, custom_palette, size, mark_size, alpha)
    return fig

def tsne_fig(
    data,
    label,
    title="t-SNE embedding",
    xlabel="tsne_x",
    ylabel="tsne_y",
    custom_palette=None,
    size=(10, 10),
    mark_size = 40,
    alpha = 0.6
):
    """Get T-SNE embeddings figure"""
    tsne_result = get_embedding(data, method = "tsne")
    fig = plot_embedding(tsne_result, label, title, xlabel,
                         ylabel, custom_palette, size, mark_size, alpha)
    return fig


def test_tsne():
    data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # data=torch.Tensor(data)
    # data=torch.Tensor(data).cuda()
    label = np.array([1, 2, 3, 4])
    fig = tsne_fig(
        data, label, title="t-SNE embedding", xlabel="tsne_x", ylabel="tsne_y"
    )
    plt.show(fig)


# https://stackoverflow.com/questions/58766561/scikit-learn-sklearn-confusion-matrix-plot-for-more-than-3-classes
def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    save_fig_path=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix

    cm = np.zeros((len(classes), len(classes)))
    for i in range(y_true.shape[0]):
        cm[y_true[i], y_pred[i]] += 1

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis]+1e-24)
        print("Normalized confusion matrix")
    else:
        cm = cm.astype("int")
        print("Confusion matrix, without normalization")

    # print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    return ax, cm


def get_class_name(dataset, num_class, args):
    if dataset == "cifar10":
        # https://www.cs.toronto.edu/~kriz/cifar.html
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif dataset == "cifar100":
        # https://github.com/keras-team/keras/issues/2653
        return [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
    elif dataset == "gtsrb":
        # https://github.com/magnusja/GTSRB-caffe-model/blob/master/labeller/main.py
        return [
            "20_speed",
            "30_speed",
            "50_speed",
            "60_speed",
            "70_speed",
            "80_speed",
            "80_lifted",
            "100_speed",
            "120_speed",
            "no_overtaking_general",
            "no_overtaking_trucks",
            "right_of_way_crossing",
            "right_of_way_general",
            "give_way",
            "stop",
            "no_way_general",
            "no_way_trucks",
            "no_way_one_way",
            "attention_general",
            "attention_left_turn",
            "attention_right_turn",
            "attention_curvy",
            "attention_bumpers",
            "attention_slippery",
            "attention_bottleneck",
            "attention_construction",
            "attention_traffic_light",
            "attention_pedestrian",
            "attention_children",
            "attention_bikes",
            "attention_snowflake",
            "attention_deer",
            "lifted_general",
            "turn_right",
            "turn_left",
            "turn_straight",
            "turn_straight_right",
            "turn_straight_left",
            "turn_right_down",
            "turn_left_down",
            "turn_circle",
            "lifted_no_overtaking_general",
            "lifted_no_overtaking_trucks",
        ]
    elif dataset == "tiny":
        path = args.dataset_path + "/tiny/tiny-imagenet-200/"
        label_map = get_class_to_id_dict(path)
        return [label_map[i][1].strip().split(",")[0] for i in range(num_class)]
    else:
        print("Class Name is not implemented currently and use label directly.")
        return [str(i) for i in range(num_class)]


def sample_by_classes(x, y, class_sub):
    sub_idx = []
    for c_idx in class_sub:
        sub_idx.append(np.where(y == c_idx))
    sub_idx = np.concatenate(sub_idx, 1).reshape(-1)
    label_sub = y[sub_idx]
    img_sub = [x[img_idx] for img_idx in sub_idx]
    return img_sub, label_sub


def sub_sample_euqal_classes(x, y, num_sample):
    class_unique = np.unique(y)
    select_idx = []
    sub_num = int(num_sample/class_unique.shape[0])
    for c_idx in class_unique:
        sub_idx = np.where(y == c_idx)
        sub_idx = np.random.choice(sub_idx[0], sub_num, replace=False)
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    # shuffle the sub_idx
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    label_sub = y[sub_idx]
    img_sub = [x[img_idx] for img_idx in sub_idx]
    return img_sub, label_sub


def sub_sample_euqal_classes_index(y, num_sample, selected_classes=None):
    # subsample the data with equal number for each classes
    class_unique = np.unique(y)
    if selected_classes is not None:
        # find the intersection of selected_classes and class_unique
        class_unique = np.intersect1d(
            class_unique, selected_classes, assume_unique=True, return_indices=False)
    select_idx = []
    sub_num = int(num_sample/class_unique.shape[0])
    for c_idx in class_unique:
        sub_idx = np.where(y == c_idx)
        sub_idx = np.random.choice(sub_idx[0], sub_num, replace=False)
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    # shuffle the sub_idx
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    return sub_idx


def sub_sample_euqal_ratio_classes_index(y, ratio=None, selected_classes=None, max_num_samples=None):
    # subsample the data with ratio for each classes
    class_unique = np.unique(y)
    if selected_classes is not None:
        # find the intersection of selected_classes and class_unique
        class_unique = np.intersect1d(
            class_unique, selected_classes, assume_unique=True, return_indices=False)
    select_idx = []
    if max_num_samples is not None:
        print('max_num_samples is given, use sample number limit now.')
        total_selected_samples = np.sum(
            [np.where(y == c_idx)[0].shape[0] for c_idx in class_unique])
        ratio = np.min([total_selected_samples, max_num_samples]
                       )/total_selected_samples

    for c_idx in class_unique:
        sub_idx = np.where(y == c_idx)
        sub_idx = np.random.choice(sub_idx[0], int(
            ratio*sub_idx[0].shape[0]), replace=False)
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    # shuffle the sub_idx
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    return sub_idx


# https://colab.research.google.com/github/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb#scrollTo=7TUH7bu7n5ta
def get_id_dictionary(path, by_wnids=False):
    if by_wnids:
        id_dict = {}
        for i, line in enumerate(open(path + "wnids.txt", "r")):
            id_dict[line.replace("\n", "")] = i
        return id_dict
    else:
        classes = sorted(
            entry.name for entry in os.scandir(path + "/train") if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {path+'/train'}."
            )
        return {cls_name: i for i, cls_name in enumerate(classes)}


def get_class_to_id_dict(path):
    id_dict = get_id_dictionary(path)
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + "words.txt", "r")):
        n_id, word = line.split("\t")[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result


def get_dataname(dataset):
    # "mnist, cifar10, cifar100, gtsrb, celeba, tiny"
    if dataset == "mnist":
        return "MNIST"
    elif dataset == 'cifar10':
        return "CIFAR-10"
    elif dataset == 'cifar100':
        return "CIFAR-100"
    elif dataset == "gtsrb ":
        return "GTSRB "
    elif dataset == "celeba":
        return "CelebA"
    elif dataset == "tiny":
        return "Tiny ImageNet"
    else:
        return dataset


def get_pratio(pratio):
    # convert 0.1 to 10% and 0.01 to 0.1%
    pratio = float(pratio)
    if pratio >= 0.1:
        return "%d" % (pratio*100)
    elif pratio >= 0.01:
        return "%d" % (pratio*100)
    elif pratio >= 0.001:
        return "%.1f" % (pratio*100)
    else:
        return "%f" % (pratio*100)


def get_defensename(defense):
    # Formal Abbreviation of Defense
    if defense == 'ft':
        return "FT"
    elif defense == 'fp':
        return "FP"
    elif defense == 'anp':
        return "ANP"
    else:
        return defense


def saliency(input, model):
    for param in model.parameters():
        param.requires_grad = False
    input.unsqueeze_(0)
    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    score.backward()
    gradients = input.grad.data.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    gradients_fre = np.fft.ifft2(gradients, axes=(0, 1))

    gradients_fre_shift = np.fft.fftshift(gradients_fre, axes=(0, 1))
    gradients_fre_shift = np.log(np.abs(gradients_fre_shift))

    gradient_norm = (gradients_fre_shift - gradients_fre_shift.min()) / \
        (gradients_fre_shift.max()-gradients_fre_shift.min())
    gradient_norm = np.mean(gradient_norm, axis=2)
    gradient_norm = np.uint8(255 * gradient_norm)
    return gradient_norm
