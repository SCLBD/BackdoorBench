import sys
import os
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms

sys.path.append(os.getcwd())

from utils.defense_utils.dbd.model.model import SelfModel, LinearModel
from utils.defense_utils.dbd.model.utils import (
    get_network_dbd,
    load_state,
    get_criterion,
    get_optimizer,
    get_scheduler,
)
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from visual_utils import *

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

# Create dataset
if args.visual_dataset == 'mixed':
    bd_test_with_trans = result_attack["bd_test"]
    visual_dataset = generate_mix_dataset(
        bd_test_with_trans, args.target_class, args.pratio, selected_classes, max_num_samples=args.n_sub)
elif args.visual_dataset == 'clean_train':
    clean_train_with_trans = result_attack["clean_train"]
    visual_dataset = generate_clean_dataset(
        clean_train_with_trans, selected_classes, max_num_samples=args.n_sub)
elif args.visual_dataset == 'clean_test':
    clean_test_with_trans = result_attack["clean_test"]
    visual_dataset = generate_clean_dataset(
        clean_test_with_trans, selected_classes, max_num_samples=args.n_sub)
elif args.visual_dataset == 'bd_train':
    bd_train_with_trans = result_attack["bd_train"]
    visual_dataset = generate_bd_dataset(
        bd_train_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub)
elif args.visual_dataset == 'bd_test':
    bd_test_with_trans = result_attack["bd_test"]
    visual_dataset = generate_bd_dataset(
        bd_test_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub)
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

############ Activation Image ################
print('Plotting Activation Image')

# Choose layer for feature extraction
module_dict = dict(model_visual.named_modules())
target_layer = module_dict[args.target_layer_name]
print(f'Choose layer {args.target_layer_name} from model {args.model}')

# Get features
features, labels, poi_indicator = get_features(args, model_visual, target_layer, data_loader, reduction='sum')
total_neuron = features.shape[1]


if args.neuron_order == 'ordered':
    target_sort = np.arange(total_neuron)
elif args.neuron_order == 'random':
    target_sort = np.random.shuffle(np.arange(total_neuron))
else:
    print(f'Illegal Neuron order: {args.neuron_order}. Use "ordered" instead')
    target_sort = np.arange(total_neuron)

# get top activation images for each Neuron
top_indx=np.argsort(-features,axis=0)

# Choose some nurons to visualize
num_neuron = np.min([args.num_neuron,total_neuron])
num_image = args.num_image
fig, axes = plt.subplots(nrows=num_neuron, ncols=num_image, figsize=(4*num_image, 5*num_neuron))
for neu_i in range(num_neuron):
    im = target_sort[neu_i]
    for topi in range(num_image):
        top_i = top_indx[topi,im]
        ax = axes[neu_i, topi]
        cnn_image = np.swapaxes(np.swapaxes(denormalizer(visual_dataset[top_i][0]).cpu().numpy(), 0, 1), 1, 2)
        cnn_image = cnn_image.clip(0,1)
        ax.imshow(cnn_image)
        if poi_indicator[top_i]==1:
            ax.set_title(f'Neuron {im}, Top-{topi}, Value {features[top_i,im]:.2f}',color = 'red')
        else:
            ax.set_title(f'Neuron {im}, Top-{topi}, Value {features[top_i,im]:.2f}',color = 'black')

plt.tight_layout()
plt.savefig(visual_save_path + f"/act_{args.visual_dataset}.png")

print(f'Save to {visual_save_path + f"/act_{args.visual_dataset}"}.png')
