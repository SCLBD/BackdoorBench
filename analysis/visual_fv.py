import sys
import os
import yaml
sys.path.append("../")
sys.path.append("../../")
sys.path.append(os.getcwd())

from PIL import Image
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
import torch
import numpy as np
import torchvision.transforms as transforms
from omnixai.explainers.vision.specific.feature_visualization.visualizer import FeatureVisualizer

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

module_dict = dict(model_visual.named_modules())
target_layer = module_dict[args.target_layer_name]
print(f'Choose layer {args.target_layer_name} from model {args.model}')

# Enable training transform to enhance transform robustness
tran = get_transform(
    args.dataset, *([args.input_height, args.input_width]), train=True)

for trans_t in tran.transforms:
    if isinstance(trans_t, transforms.Normalize):
        denormalizer = get_dataset_denormalization(trans_t)

############## Feature Visualization ##################
print("Plotting Feature Visualization")

optimizer = FeatureVisualizer(
    model = model_visual,
    objectives = [{"layer": target_layer, "type": "channel",
                 "index": list(range(target_layer.out_channels))}],
    transformers = tran
)

# Some regularizations are used for better visualization results.
# The parameter for regularization is self-defined and you should set them by yourself.
# Note that such regularization may hinder optimizer to find some triggers especially when the triggers are some irregular patterns.
explanations = optimizer.explain(
    num_iterations=300,
    image_shape=(args.input_height, args.input_width),
    regularizers=[("l1", 0.15), ("l2", 0), ("tv", 0.25)],
    use_fft=True,
)

images = explanations.explanations[0]['image']
num_cnn = len(images)
num_col = 16
num_row = int(np.ceil(num_cnn/num_col))
fig, axes = plt.subplots(nrows=num_row, ncols=num_col,
                         figsize=(4*num_col, 5*num_row))
for cnn_i in range(num_cnn):
    ax = axes[cnn_i//num_col, cnn_i % num_col]
    ax.imshow(images[cnn_i])
    ax.set_title(f'Kernel {cnn_i}')

plt.tight_layout()
plt.savefig(visual_save_path + f"/feature_visual.png")

print(f'Save to {visual_save_path + f"/feature_visual"}.png')
