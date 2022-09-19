import os
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_args():
    # set the basic parameter
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="cuda, cpu")
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./config/visualization/default.yaml",
        help="the path of yaml",
    )
    parser.add_argument("--seed", type=str, help="random seed")
    parser.add_argument("--model", type=str, help="preactresnet18")

    # data parameters
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument(
        "--dataset", type=str, help="mnist, cifar10, cifar100, gtsrb, celeba, tiny"
    )
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_sub", default=5000, type=int)
    parser.add_argument("--c_sub", default=10, type=int)
    parser.add_argument("--num_workers", default=2, type=float)
    parser.add_argument("--class_names", type=list,
                        help="names for each class")

    # BD parameters
    parser.add_argument("--pratio", type=float)
    parser.add_argument("--attack_target", type=int)
    parser.add_argument("--index", type=str, help="index of clean data")

    # results parameters
    parser.add_argument(
        "--result_file_attack",
        default=None,
        type=str,
        help="the location of attack result",
    )
    parser.add_argument(
        "--result_file_defense",
        default=None,
        type=str,
        help="the location of defense result",
    )
    parser.add_argument("--checkpoint_load", default=None, type=str)
    parser.add_argument("--checkpoint_save", default=None, type=str)

    arg = parser.parse_args()
    return arg


def preprocess_args(args):
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

    return args


def get_features(args, model, target_layer, data_loader):
    '''Function to extract the features/embeddings/activations from a target layer'''

    # extract feature vector from a specific layer
    def feature_hook(module, input_, output_):
        global feature_vector
        # access the layer output and convert it to a feature vector
        feature_vector = torch.flatten(output_, 1)
        return None

    h = target_layer.register_forward_hook(feature_hook)

    model.eval()
    # collect feature vectors
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            global feature_vector
            # Fetch features
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            current_feature = feature_vector.cpu().numpy()
            current_labels = targets.cpu().numpy()

            # Store features
            features.append(current_feature)
            labels.append(current_labels)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    h.remove()  # Rmove the hook

    return features, labels


def plot_embedding(
    tsne_result, label, title, xlabel="tsne_x", ylabel="tsne_y", custom_palette=None, size=(10, 10)
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
        {"tsne_x": tsne_result[:, 0],
            "tsne_y": tsne_result[:, 1], "label": label}
    )
    fig, ax = plt.subplots(1, figsize=size)

    num_class = len(pd.unique(tsne_result_df["label"]))
    if custom_palette is None:
        custom_palette = sns.color_palette("hls", num_class)

    # s: maker size, palette: colors

    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue="label",
        data=tsne_result_df,
        ax=ax,
        s=40,
        palette=custom_palette,
        alpha=0.6,
    )
    #     sns.lmplot(x='tsne_x', y='tsne_y', hue='label',
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


def get_embedding(data):
    """Get T-SNE embeddings"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    result = tsne.fit_transform(data)
    return result


def tsne_fig(
    data,
    label,
    title="t-SNE embedding",
    xlabel="tsne_x",
    ylabel="tsne_y",
    custom_palette=None,
    size=(10, 10)
):
    """Get T-SNE embeddings figure"""
    tsne_result = get_embedding(data)
    fig = plot_embedding(tsne_result, label, title, xlabel,
                         ylabel, custom_palette, size)
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
