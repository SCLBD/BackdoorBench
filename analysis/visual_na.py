import sys
import os
sys.path.append("../")
sys.path.append(os.getcwd())

from visual_utils import *
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_attack_result
from utils.defense_utils.dbd.model.utils import (
    get_network_dbd,
    load_state,
    get_criterion,
    get_optimizer,
    get_scheduler,
)
from utils.defense_utils.dbd.model.model import SelfModel, LinearModel
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms


# Basic setting: args
args = get_args()

with open(args.yaml_path, "r") as stream:
    config = yaml.safe_load(stream)
config.update({k: v for k, v in args.__dict__.items() if v is not None})
args.__dict__ = config
args = preprocess_args(args)
fix_random(int(args.random_seed))

save_path_attack = "./record/" + args.result_file_attack
visual_save_path = save_path_attack + "/visual"


# Load result
if args.prototype:
    result_attack = load_prototype_result(args, save_path_attack)
else:
    result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

selected_classes = np.arange(args.num_classes)

# Select classes to visualize
if args.num_classes > args.c_sub:
    selected_classes = np.delete(selected_classes, args.target_class)
    selected_classes = np.random.choice(
        selected_classes, args.c_sub-1, replace=False)
    selected_classes = np.append(selected_classes, args.target_class)

# keep the same transforms for train and test dataset for better visualization
result_attack["clean_train"].wrap_img_transform = result_attack["clean_test"].wrap_img_transform 
result_attack["bd_train"].wrap_img_transform = result_attack["bd_test"].wrap_img_transform 

# Create dataset. Only support BD_TEST and BD_TRAIN
if args.visual_dataset == 'bd_train':
    bd_train_with_trans = result_attack["bd_train"]
    visual_dataset = generate_bd_dataset(
        bd_train_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub, bd_only=True)
elif args.visual_dataset == 'bd_test':
    bd_test_with_trans = result_attack["bd_test"]
    visual_dataset = generate_bd_dataset(
        bd_test_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub, bd_only=True)
else:
    assert False, "Illegal vis_class"

print(
    f'Create visualization dataset with \n \t Dataset: {args.visual_dataset} \n \t Number of samples: {len(visual_dataset)}  \n \t Selected classes: {selected_classes}')

# Create data loader
data_loader = torch.utils.data.DataLoader(
    visual_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)

# Create denormalization function
for trans_t in data_loader.dataset.wrap_img_transform.transforms:
    if isinstance(trans_t, transforms.Normalize):
        denormalizer = get_dataset_denormalization(trans_t)

# Load model
model_visual = generate_cls_model(args.model, args.num_classes)

if args.result_file_defense != "None":
    save_path_defense = "./record/" + args.result_file_defense
    visual_save_path = save_path_defense + "/visual"

    result_defense = load_attack_result(
        save_path_defense + "/defense_result.pt")
    defense_method = args.result_file_defense.split('/')[-1]
    if defense_method == 'fp':
        model_visual.layer4[1].conv2 = torch.nn.Conv2d(
            512, 512 - result_defense['index'], (3, 3), stride=1, padding=1, bias=False)
        model_visual.linear = torch.nn.Linear(
            (512 - result_defense['index'])*1, args.num_classes)
    if defense_method == 'dbd':
        backbone = get_network_dbd(args)
        model_visual = LinearModel(
            backbone, backbone.feature_dim, args.num_classes)
    model_visual.load_state_dict(result_defense["model"])
    print(f"Load model {args.model} from {args.result_file_defense}")
else:
    model_visual.load_state_dict(result_attack["model"])
    print(f"Load model {args.model} from {args.result_file_attack}")

model_visual.to(args.device)

# !!! Important to set eval mode !!!
model_visual.eval()

# make visual_save_path if not exist
os.mkdir(visual_save_path) if not os.path.exists(visual_save_path) else None

############## Neuron Activation ##################
print("Plotting Neuron Activation")

# Choose layer for feature extraction
module_dict = dict(model_visual.named_modules())
target_layer = module_dict[args.target_layer_name]
print(f'Choose layer {args.target_layer_name} from model {args.model}')

# Get BD features
features_bd, labels_bd, other_info = get_features(
    args, model_visual, target_layer, data_loader)
features_bd_avg = np.mean(features_bd, axis=0)

# Get Corresponding Clean features
visual_dataset.wrapped_dataset.poison_indicator = np.zeros_like(
    visual_dataset.wrapped_dataset.poison_indicator)

features_clean, labels_clean, other_info = get_features(
    args, model_visual, target_layer, data_loader)
features_clean_avg = np.mean(features_clean, axis=0)

sort_bar = np.argsort(features_clean_avg)[::-1]

features_bd_avg = features_bd_avg[sort_bar]
features_clean_avg = features_clean_avg[sort_bar]

plt.figure(figsize=(10, 10))
plt.bar(
    np.arange(features_clean_avg.shape[0]),
    features_clean_avg,
    label="Clean",
    alpha=0.7,
    color="#2196F3",
)
plt.xlabel("Neuron")
plt.ylabel("Average Activation Value")
plt.title(
    f"{get_dataname(args.dataset)}, {get_pratio(args.pratio)}% Poisoned Samples")
plt.xlim(0, features_clean_avg.shape[0])
plt.legend()
plt.tight_layout()
plt.savefig(visual_save_path + "/NA_clean.png")


plt.figure(figsize=(10, 10))
plt.bar(
    np.arange(features_bd_avg.shape[0]),
    features_bd_avg,
    label="Poisoned",
    alpha=0.7,
    color="#4CAF50",
)
plt.xlabel("Neuron")
plt.ylabel("Average Activation Value")
plt.title(
    f"{get_dataname(args.dataset)}, {get_pratio(args.pratio)}% Poisoned Samples")
plt.xlim(0, features_bd_avg.shape[0])
plt.legend()
plt.tight_layout()
plt.savefig(visual_save_path + "/NA_BD.png")


plt.figure(figsize=(10, 10))
plt.bar(
    np.arange(features_clean_avg.shape[0]),
    features_clean_avg,
    label="Clean",
    alpha=0.7,
    color="#2196F3",
)
plt.bar(
    np.arange(features_bd_avg.shape[0]),
    features_bd_avg,
    label="Poisoned",
    alpha=0.7,
    color="#4CAF50",
)
plt.xlabel("Neuron")
plt.ylabel("Average Activation Value")
plt.title(
    f"{get_dataname(args.dataset)}, {get_pratio(args.pratio)}% Poisoned Samples")
plt.xlim(0, features_clean_avg.shape[0])
plt.legend()

plt.tight_layout()
plt.savefig(visual_save_path + f"/NA_compare_{args.visual_dataset}.png")

print(f'Save to {visual_save_path + f"/NA_compare_{args.visual_dataset}"}.png')
