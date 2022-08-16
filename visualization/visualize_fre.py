import random
import sys, os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib as mlp
sys.path.append("../")
sys.path.append(os.getcwd())
from visual_utils import *
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result

### 1. basic setting: args
args = get_args()
with open(args.yaml_path, "r") as stream:
    config = yaml.safe_load(stream)
config.update({k: v for k, v in args.__dict__.items() if v is not None})
args.__dict__ = config
args = preprocess_args(args)
fix_random(int(args.random_seed))
save_path_attack = "./record/" + args.result_file_attack
if args.result_file_defense!='None':
    save_path = "./record/" + args.result_file_defense
else:
    save_path = save_path_attack
args.save_path = save_path

# Load data
result_attack = load_attack_result(save_path_attack + "/attack_result.pt")
x = result_attack["clean_test"]["x"]
y = result_attack["clean_test"]["y"]
x_bd = result_attack["bd_test"]["x"]
y_bd = result_attack["bd_test"]["y"]

# select index
x_bd_ori_idx = result_attack['bd_test']['original_index']
index = random.sample(np.arange(len(x_bd)).tolist(), 2)
index_ori = [x_bd_ori_idx[i] for i in index]

x_benign_select = []
y_benign_select = []
x_bd_select = []
y_bd_select = []
for id in index_ori:
    x_benign_select.append(np.array(x[id]))
    y_benign_select.append(np.array(y[id]))
for id in index:
    x_bd_select.append(np.array(x_bd[id]))
    y_bd_select.append(np.array(y_bd[id]))

if args.result_file_defense!='None':
    result = load_attack_result(save_path + "/defense_result.pt")
else:
    result = result_attack

# Load model
model = generate_cls_model(args.model, args.num_classes)
model.load_state_dict(result["model"])
model.to(args.device)

criterion = nn.CrossEntropyLoss()
tran = get_transform(
    args.dataset, *([args.input_height, args.input_width]), train=False
)

x_v = np.concatenate((x_benign_select,x_bd_select), axis=0)
y_v = np.concatenate((y_benign_select,y_bd_select), axis=0)
data_set = list(zip(x_v, y_v))
data_set_o = prepro_cls_DatasetBD(
    full_dataset_without_transform=data_set,
    poison_idx=np.zeros(
        len(data_set)
    ),  # no further poison steps for visualization data
    bd_image_pre_transform=None,
    bd_label_pre_transform=None,
    ori_image_transform_in_loading=tran,
    ori_label_transform_in_loading=None,
    add_details_in_preprocess=False,
)
data_loader = torch.utils.data.DataLoader(
    data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)

for trans_t in data_loader.dataset.ori_image_transform_in_loading.transforms:
    if isinstance(trans_t, transforms.Normalize):
        denormalizer = get_dataset_denormalization(trans_t)
class_names = np.array(args.class_names).reshape([-1])
test_label = np.concatenate([y_v[:2], y_v[:2]])
############ Frequency saliency map ################
full_image = []
full_target = []
for batch_idx, (inputs, targets) in enumerate(data_loader):
    full_image.append(inputs)
    full_target.append(targets)
full_image_t = torch.cat(full_image, 0)
full_target_t = torch.cat(full_target, 0)
test_images = torch.cat([full_image_t[:2], full_image_t[-2:]]).to(args.device)
sfm = nn.Softmax(dim=1)
outputs = model(test_images)
pre_p, pre_label = torch.max(sfm(outputs), dim=1)

print('Plotting frequency saliency map')
if args.model == "preactresnet18":
    target_layers = [model.layer4]
if args.model == "vgg19":
    target_layers = [model.features]
if args.model == "resnet18":
    target_layers = [model.layer4]
if args.model == "densenet161":
    target_layers = [model.features]
if args.model == "mobilenet_v3_large":
    target_layers = [model.features]
if args.model == "efficientnet_b3":
    target_layers = [model.features]

model.eval()
frequency_maps = []
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
vnorm = mlp.colors.Normalize(vmin=0, vmax=255)
for im in range(4):
    rgb_image = np.swapaxes(
        np.swapaxes(denormalizer(test_images[im]).cpu().numpy(), 0, 1), 1, 2
    )
    frequency_map = saliency(test_images[im], model)
    rgb_image[rgb_image < 1e-12] = 1e-12
    axes[im // 2, im % 2 * 2].imshow(rgb_image)
    axes[im // 2, im % 2 * 2].axis("off")
    if im == 0 or im ==1:
        axes[im // 2, im % 2 * 2].set_title(
            "Clean Image: %s" % (class_names[test_label[im]].capitalize())
        )
    else:
        axes[im // 2, im % 2 * 2].set_title(
            "Poison Image: %s" % (class_names[test_label[im]].capitalize())
        )
    image = axes[im // 2, im % 2 * 2 + 1].imshow(frequency_map, cmap=plt.cm.coolwarm, norm=vnorm)
    plt.colorbar(image, ax=axes[im // 2, im % 2 * 2 + 1], orientation='vertical')
    axes[im // 2, im % 2 * 2 + 1].axis("off")
    axes[im // 2, im % 2 * 2 + 1].set_title(
        "Predicted: %s, %.2f%%" % (class_names[pre_label[im]].capitalize(), pre_p[im] * 100)
    )
plt.savefig(args.save_path + "/frequency_map.png")