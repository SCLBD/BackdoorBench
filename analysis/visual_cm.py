import sys
import os
sys.path.append("../")
sys.path.append(os.getcwd())

from matplotlib.patches import Rectangle, Patch
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

############## Confusion Matrix ##################
print("Plotting Confusion Matrix")

target_class = args.target_class
poison_class = args.num_classes
class_names = args.class_names

# Evaluation
criterion = torch.nn.CrossEntropyLoss()
total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
target_correct, target_total = 0, 0

true_labls = []
pred_labels = []
for i, (inputs, labels, *other_info) in enumerate(data_loader):
    inputs, labels = inputs.to(args.device), labels.to(args.device)
    outputs = model_visual(inputs)
    loss = criterion(outputs, labels)
    test_loss += loss.item()

    total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
    target_correct += torch.sum(
        (torch.argmax(outputs[:], dim=1) == target_class) * (labels[:] == target_class)
    )
    target_total += torch.sum(labels[:] == target_class)

    total_clean_test += inputs.shape[0]
    avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
    prediction = torch.argmax(outputs[:], dim=1)
    true_labls.append(labels.detach().cpu().numpy())
    pred_labels.append(prediction.detach().cpu().numpy())
    
true_labls = np.concatenate(true_labls)
pred_labels = np.concatenate(pred_labels)

plot_confusion_matrix(
    true_labls,
    pred_labels,
    classes=class_names,
    normalize=True,
    title="Confusion matrix",
    save_fig_path=None,
)

plt.tight_layout()
plt.savefig(visual_save_path + f"/cm_{args.visual_dataset}.png")

print(f'Save to {visual_save_path + f"/cm_{args.visual_dataset}"}.png')

print(
    "Acc: {:.3f}%({}/{})".format(
        avg_acc_clean, total_clean_correct_test, total_clean_test
    )
)
print(
    "Acc (Target only): {:.3f}%({}/{})".format(
        target_correct / target_total * 100.0, target_correct, target_total
    )
)
