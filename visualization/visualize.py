import sys, os
import yaml
import shap
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

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
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

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
# original label of bd data
y_bd_clean = y[y != args.attack_target]
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

# subsample classes for visualization if necessary
if args.num_classes > args.c_sub:
    # subsample c_sub classes
    class_sub = 1 + np.random.choice(
        args.num_classes - 1, size=args.c_sub - 1, replace=False
    )
    class_sub = class_sub.tolist()
    class_sub.append(args.attack_target)
    x, y = sample_by_classes(x, y, class_sub)
    x_bd, y_bd = sample_by_classes(x_bd, y_bd_clean, class_sub)


# Subsample data

n = len(x)
n_sub = args.n_sub
# subsample clean data
if n > n_sub:
    sub_bin = int(n / n_sub)
    sub_idx = np.argsort(y)[::sub_bin]
    y_sub = y[sub_idx]
    x_sub = [x[i] for i in sub_idx]
else:
    y_sub = y
    x_sub = x

# subsample poison data
n_sub_bd = int(len(x_sub) * args.pratio)
sub_bin_bd = int(n / n_sub_bd)
sub_idx_bd = np.argsort(y_bd)[::sub_bin_bd]
y_bd_sub = y_bd[sub_idx_bd]

y_bd_clean_sub = np.copy(y_bd_sub)
y_bd_sub[:] = args.num_classes

x_bd_sub = [x_bd[i] for i in sub_idx_bd]

x_v = x_sub + x_bd_sub
y_v = np.concatenate([y_sub, y_bd_sub])
y_v_clean = np.concatenate([y_sub, y_bd_clean_sub])

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


############## T-SNE ##################
print('Plotting T-SNE')
# Choose layer for feature extraction
if args.model == "preactresnet18":
    target_layer = model.layer4
if args.model == "vgg19":
    target_layer = model.features[-1]
if args.model == "resnet18":
    target_layer = model.layer4
if args.model == "densenet161":
    target_layer = model.features[-1]
if args.model == "mobilenet_v3_large":
    target_layer = model.features[-1]
if args.model == "efficientnet_b3":
    target_layer = model.features[-1]

# extract feature vector from a specific layer
def feature_hook(module, input_, output_):
    global feature_vector
    # access the layer output and convert it to a feature vector
    feature_vector = torch.flatten(output_, 1)
    return None


target_layer.register_forward_hook(feature_hook)

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

sort_idx = np.argsort(labels)
features = features[sort_idx]
labels = labels[sort_idx]
classes = args.class_names + ["poisoned"]
label_class = [classes[i].capitalize() for i in labels]

# Plot T-SNE
custom_palette = sns.color_palette("hls", 10) + [
    (0.0, 0.0, 0.0)
]  # Black for poison samples
fig = tsne_fig(
    features,
    label_class,
    title="t-SNE embedding",
    xlabel="tsne_x",
    ylabel="tsne_y",
    custom_palette=custom_palette,
)
plt.savefig(args.save_path + "/tsne.png")


############# SHAP Values ####################
# Choose layer for SHAP
if args.model == "preactresnet18":
    target_layer = model.layer4
if args.model == "vgg19":
    target_layer = model.features
if args.model == "resnet18":
    target_layer = model.layer4
if args.model == "densenet161":
    target_layer = model.features
if args.model == "mobilenet_v3_large":
    target_layer = model.features
if args.model == "efficientnet_b3":
    target_layer = model.features

print('Plotting SHAP Values')
model.eval()

full_image = []
full_target = []
for batch_idx, (inputs, targets) in enumerate(data_loader):
    full_image.append(inputs)
    full_target.append(targets)
full_image_t = torch.cat(full_image, 0)
full_target_t = torch.cat(full_target, 0)

# choose data for visualization
n_v = 200
v_idx = np.random.choice(full_image_t.shape[0] - 4, n_v, replace=False)
background = full_image_t[2 + v_idx].to(args.device)
# background = full_image_t[-200:].to(args.device)

# 2 clean image + 2 poisoned image
test_images = torch.cat([full_image_t[:2], full_image_t[-2:]]).to(args.device)
# Get clean label for all test image
test_label = np.concatenate([y_v_clean[:2], y_v_clean[-2:]])

sfm = nn.Softmax(dim=1)
outputs = model(test_images)
pre_p, pre_label = torch.max(sfm(outputs), dim=1)


e = shap.GradientExplainer((model, target_layer), background, local_smoothing=0)
shap_values, indexes = e.shap_values(test_images, ranked_outputs=5)

# get the names for the classes
class_names = np.array(args.class_names).reshape([-1])
index_names = np.vectorize(lambda x: class_names[x].capitalize())(indexes.cpu())
# plot the explanations
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(
    np.swapaxes(denormalizer(test_images.cpu()).numpy(), 1, -1), 1, 2
)
test_numpy[test_numpy < 1e-12] = 1e-12  # for some numerical issue

shap.image_plot(shap_numpy, test_numpy, index_names, show=False)
plt.savefig(args.save_path + "/shap.png")


############ Grad Cam ################
# choose layers for Grad Cam, refer to https://github.com/jacobgil/pytorch-grad-cam
print('Plotting Grad Cam')
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

input_tensor = test_images

cam = FullGrad(model=model, target_layers=target_layers, use_cuda=True)

targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam_full = cam(input_tensor=input_tensor, targets=targets)

grayscale_cam = grayscale_cam_full[0, :]
rgb_image = np.swapaxes(
    np.swapaxes(denormalizer(test_images[0]).cpu().numpy(), 0, 1), 1, 2
)
visual_cam = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for im in range(4):
    grayscale_cam = grayscale_cam_full[im, :]
    rgb_image = np.swapaxes(
        np.swapaxes(denormalizer(test_images[im]).cpu().numpy(), 0, 1), 1, 2
    )
    rgb_image[rgb_image < 1e-12] = 1e-12
    visual_cam = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    axes[im // 2, im % 2 * 2].imshow(rgb_image)
    axes[im // 2, im % 2 * 2].axis("off")
    axes[im // 2, im % 2 * 2].set_title(
        "Original Image: %s" % (class_names[test_label[im]].capitalize())
    )
    axes[im // 2, im % 2 * 2 + 1].imshow(visual_cam)
    axes[im // 2, im % 2 * 2 + 1].axis("off")
    axes[im // 2, im % 2 * 2 + 1].set_title(
        "Predicted: %s, %.2f%%" % (class_names[pre_label[im]].capitalize(), pre_p[im] * 100)
    )
plt.savefig(args.save_path + "/gracam.png")
