import sys
import os
sys.path.append("../")
sys.path.append("../../")
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
import yaml
import torch
import shap
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

# Prepare background data
num_bg = 200
background_idx = np.random.choice(len(visual_dataset), num_bg, replace=False)
background_samples = []
for i in background_idx:
    background_samples.append(visual_dataset[i][0].unsqueeze(0))

background_samples = torch.cat(background_samples, axis=0).to(args.device)

# Choose samples to show SHAP values. By Default, 2 clean images + 2 Poison images. If no enough Poison images, use 4 clean images instead.AblationCAM
total_num = 4
bd_num = 0

visual_samples = []
visual_labels = []

visual_poison_indicator = np.array(
    get_poison_indicator_from_bd_dataset(visual_dataset))
if visual_poison_indicator.sum() > 0:
    print(f'Number Poisoned samples: {visual_poison_indicator.sum()}')
    # random choose two poisoned samples
    selected_bd_idx = np.random.choice(
        np.where(visual_poison_indicator == 1)[0], 2, replace=False)
    for i in selected_bd_idx:
        visual_samples.append(visual_dataset[i][0].unsqueeze(0))
        visual_labels.append(visual_dataset[i][4])
    bd_num = len(selected_bd_idx)
    print(f'Select {bd_num} poisoned samples')

# Trun all samples to clean
with temporary_all_clean(visual_dataset):
    # you can just set selected_clean_idx = selected_bd_idx to build the correspondence between clean samples and poisoned samples
    selected_clean_idx = np.random.choice(
        len(visual_dataset), total_num-bd_num, replace=False)
    for i in selected_clean_idx:
        visual_samples.append(visual_dataset[i][0].unsqueeze(0))
        visual_labels.append(visual_dataset[i][1])
    print(f'Select {len(selected_clean_idx)} clean samples')

# Clean sample first
visual_samples = visual_samples[::-1]
visual_labels = visual_labels[::-1]

visual_samples = torch.cat(visual_samples, axis=0).to(args.device)

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

############## SHAP ##################
print('Plotting SHAP')

# Choose layer for feature extraction
module_dict = dict(model_visual.named_modules())
target_layer = module_dict[args.target_layer_name]
print(f'Choose layer {args.target_layer_name} from model {args.model}')

sfm = torch.nn.Softmax(dim=1)
outputs = model_visual(visual_samples)
pre_p, pre_label = torch.max(sfm(outputs), dim=1)

e = shap.GradientExplainer(
    (model_visual, target_layer), background_samples, local_smoothing=0)
shap_values, indexes = e.shap_values(visual_samples, ranked_outputs=5)

# get the names for the classes
class_names = np.array(args.class_names).reshape([-1])
index_names = np.vectorize(
    lambda x: class_names[x].capitalize())(indexes.cpu())
# plot the explanations
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(
    np.swapaxes(denormalizer(visual_samples.cpu()).numpy(), 1, -1), 1, 2
)
test_numpy[test_numpy < 1e-12] = 1e-12  # for some numerical issue

shap.image_plot(shap_numpy, test_numpy, index_names, show=False)

# plt.tight_layout() is not working
plt.savefig(visual_save_path + f"/shap_{args.visual_dataset}.png")

print(f'Save to {visual_save_path + f"/shap_{args.visual_dataset}"}.png')
