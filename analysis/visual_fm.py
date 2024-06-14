import sys
import os
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from omnixai.explainers.vision.specific.feature_visualization.visualizer import \
    FeatureMapVisualizer

sys.path.append("../")
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


# Select all classes and all samples
selected_classes = np.arange(args.num_classes)

# keep the same transforms for train and test dataset for better visualization
result_attack["clean_train"].wrap_img_transform = result_attack["clean_test"].wrap_img_transform 
result_attack["bd_train"].wrap_img_transform = result_attack["bd_test"].wrap_img_transform 

# Create dataset
if args.visual_dataset == 'clean_train':
    visual_dataset = result_attack["clean_train"]
elif args.visual_dataset == 'clean_test':
    visual_dataset = result_attack["clean_test"]
elif args.visual_dataset == 'bd_train':  
    visual_dataset = result_attack["bd_train"]
elif args.visual_dataset == 'bd_test':
    visual_dataset = result_attack["bd_test"]
else:
    assert False, "Illegal vis_class"

print(f'Create visualization dataset with \n \t Dataset: {args.visual_dataset} \n \t Number of samples: {len(visual_dataset)}  \n \t Selected classes: {selected_classes}')


# Create denormalization function
for trans_t in visual_dataset.wrap_img_transform.transforms:
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

# Choose a image to get feature maps from a target layer
module_dict = dict(model_visual.named_modules())
target_layer = module_dict[args.target_layer_name]

target_image_index = np.random.randint(0, len(visual_dataset))
target_image = visual_dataset[target_image_index][0].unsqueeze(0)
print(f"Choose image index {target_image_index} from {args.visual_dataset} as target image")

############## Feature Maps ##################
print("Plotting Feature Maps")

explainer = FeatureMapVisualizer(
    model=model_visual,
    target_layer=target_layer,
    preprocess_function=lambda x:x
)


feature_map = explainer.extractor.extract(target_image)
num_cnn = feature_map.shape[-1]
num_col = 16
num_row = int(np.ceil(num_cnn/num_col))
fig, axes = plt.subplots(nrows=num_row, ncols=num_col, figsize=(4*num_col, 5*num_row))
for cnn_i in range(num_cnn):
    ax = axes[cnn_i//num_col, cnn_i%num_col]
    ax.imshow(feature_map[0, :, :, cnn_i], cmap='gray')
    ax.set_title(f'Kernel {cnn_i}')

plt.tight_layout()
plt.savefig(visual_save_path + f"/feature_map_{args.visual_dataset}.png")

print(f'Save to {visual_save_path + f"/feature_map_{args.visual_dataset}"}.png')
