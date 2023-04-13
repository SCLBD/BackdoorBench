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

import matplotlib.pyplot as plt
from utils.metric import *
import warnings

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
visual_dataset_clean = result_attack["clean_test"]
visual_dataset_bd = result_attack["bd_test"]

print(f'Create clean test dataset with {len(visual_dataset_clean)} samples')
print(f'Create poison test dataset with {len(visual_dataset_bd)} samples')

# Create data loader
data_loader_clean = torch.utils.data.DataLoader(
    visual_dataset_clean, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)

data_loader_bd = torch.utils.data.DataLoader(
    visual_dataset_bd, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)


metric_dic = {}
# Load model
model_attack = generate_cls_model(args.model, args.num_classes)
model_defense = None
model_attack.load_state_dict(result_attack["model"])
print(f"Load model {args.model} from {args.result_file_attack}")
model_attack.to(args.device)
model_attack.eval()


if args.result_file_defense != "None":
    model_defense = generate_cls_model(args.model, args.num_classes)
    save_path_defense = "./record/" + args.result_file_defense
    visual_save_path = save_path_defense + "/visual"

    result_defense = load_attack_result(
        save_path_defense + "/defense_result.pt")
    defense_method = args.result_file_defense.split('/')[-1]
    if defense_method == 'fp':
        model_defense.layer4[1].conv2 = torch.nn.Conv2d(
            512, 512 - result_defense['index'], (3, 3), stride=1, padding=1, bias=False)
        model_defense.linear = torch.nn.Linear(
            (512 - result_defense['index'])*1, args.num_classes)
    if defense_method == 'dbd':
        backbone = get_network_dbd(args)
        model_defense = LinearModel(
            backbone, backbone.feature_dim, args.num_classes)
    model_defense.load_state_dict(result_defense["model"])
    print(f"Load model {args.model} from {args.result_file_defense}")
    model_defense.to(args.device)
    model_defense.eval()


# make visual_save_path if not exist
os.mkdir(visual_save_path) if not os.path.exists(visual_save_path) else None


target_class = args.target_class
poison_class = args.num_classes
class_names = args.class_names

############## Collect Attack Predicts ##################
print("Collecting attack predicts")

# Evaluation
# Clean part
true_labels_clean_attack = []
pred_labels_clean_attack = []
true_labels_clean_defense = []
pred_labels_clean_defense = []

for i, (inputs, labels, *other_info) in enumerate(data_loader_clean):
    inputs, labels = inputs.to(args.device), labels.to(args.device)
    
    # attack part
    outputs = model_attack(inputs)
    prediction = torch.argmax(outputs[:], dim=1)
    true_labels_clean_attack.append(labels.detach().cpu().numpy())
    pred_labels_clean_attack.append(prediction.detach().cpu().numpy())
    
    # defense part
    if model_defense is not None:
        outputs = model_defense(inputs)
        prediction = torch.argmax(outputs[:], dim=1)
        true_labels_clean_defense.append(labels.detach().cpu().numpy())
        pred_labels_clean_defense.append(prediction.detach().cpu().numpy())

true_labels_clean_attack = np.concatenate(true_labels_clean_attack)
pred_labels_clean_attack = np.concatenate(pred_labels_clean_attack)

if model_defense is not None:
    true_labels_clean_defense = np.concatenate(true_labels_clean_defense)
    pred_labels_clean_defense = np.concatenate(pred_labels_clean_defense)

# clean accuracy
clean_accuracy_attack = clean_accuracy(pred_labels_clean_attack, true_labels_clean_attack)
metric_dic['clean_accuracy_attack'] = clean_accuracy_attack
if model_defense is not None:
    clean_accuracy_defense = clean_accuracy(pred_labels_clean_defense, true_labels_clean_defense)
    metric_dic['clean_accuracy_defense'] = clean_accuracy_defense


# Backdoor part
true_labels_bd_attack = []
pred_labels_bd_attack = []
ori_labels_bd_attack = []

true_labels_bd_defense = []
pred_labels_bd_defense = []
ori_labels_bd_defense = []

for i, (inputs, labels, *other_info) in enumerate(data_loader_bd):
    inputs, labels = inputs.to(args.device), labels.to(args.device)
    
    if torch.sum(other_info[1]==0)>0:
        # warning message
        warnings.warn("There are some clean samples in backdoor dataset detected by the poison indicators. Please Check you dataset.")

    # attack part
    outputs = model_attack(inputs)
    prediction = torch.argmax(outputs[:], dim=1)
    true_labels_bd_attack.append(labels.detach().cpu().numpy())
    pred_labels_bd_attack.append(prediction.detach().cpu().numpy())
    ori_labels_bd_attack.append(other_info[2].detach().cpu().numpy())
    
    # defense part
    if model_defense is not None:
        outputs = model_defense(inputs)
        prediction = torch.argmax(outputs[:], dim=1)
        true_labels_bd_defense.append(labels.detach().cpu().numpy())
        pred_labels_bd_defense.append(prediction.detach().cpu().numpy())
        ori_labels_bd_defense.append(other_info[2].detach().cpu().numpy())

true_labels_bd_attack = np.concatenate(true_labels_bd_attack)
pred_labels_bd_attack = np.concatenate(pred_labels_bd_attack)
ori_labels_bd_attack = np.concatenate(ori_labels_bd_attack)

if model_defense is not None:
    true_labels_bd_defense = np.concatenate(true_labels_bd_defense)
    pred_labels_bd_defense = np.concatenate(pred_labels_bd_defense)
    ori_labels_bd_defense = np.concatenate(ori_labels_bd_defense)



# attack success rate
asr_attack = attack_success_rate(pred_labels_bd_attack, true_labels_bd_attack)
metric_dic['asr_attack'] = asr_attack

ra_attack = robust_accuracy(pred_labels_bd_attack, ori_labels_bd_attack)
metric_dic['ra_attack'] = ra_attack

if model_defense is not None:
    asr_defense = attack_success_rate(pred_labels_bd_defense, true_labels_bd_defense)
    metric_dic['asr_defense'] = asr_defense
    
    ra_defense = robust_accuracy(pred_labels_bd_defense, ori_labels_bd_defense)
    metric_dic['ra_defense'] = ra_defense
    
if model_defense is not None:
    # Assume the original label and the true label are shared by both attack and defense
    der = defense_effectiveness_rate_simplied(clean_accuracy_attack, clean_accuracy_defense, asr_attack, asr_defense)
    rir = robust_improvement_rate_simplied(clean_accuracy_attack, clean_accuracy_defense, ra_attack, ra_defense)
    
    metric_dic['der'] = der
    metric_dic['rir'] = rir
    
# print metric
for key, value in metric_dic.items():
    print(f"{key}: {value}")

summary = pd.DataFrame(metric_dic, index=[0])
summary.to_csv(f'{visual_save_path}/metric_summary.csv', index=False)

print(f'Save to {visual_save_path}/metric_summary.csv')


### Visualization
metric_2_name = {'clean_accuracy_attack': 'C-ACC', 'clean_accuracy_defense': 'C-ACC', 'asr_attack': '1 - ASR', 'asr_defense': '1 - ASR', 'ra_attack': 'RA', 'ra_defense': 'RA', 'der': 'DER', 'rir': 'RIR'}
if model_defense is not None:
    used_metrics = ['clean_accuracy_defense', 'asr_defense', 'ra_defense', 'der', 'rir']
    if 'asr_defense' in used_metrics:
        metric_dic['asr_defense'] = 1 - metric_dic['asr_defense']
        print('Turn ASR to 1-ASR for visualization.')
    plot_metrics = [metric_2_name[key] for key in used_metrics]
    plot_metrics_values = [metric_dic[key] for key in used_metrics]
else:
    used_metrics = ['clean_accuracy_attack', 'asr_attack', 'ra_attack']
    if 'asr_attack' in used_metrics:
        metric_dic['asr_attack'] = 1 - metric_dic['asr_attack']
        print('Turn ASR to 1-ASR for visualization.')
    plot_metrics = [metric_2_name[key] for key in used_metrics]
    plot_metrics_values = [metric_dic[key] for key in used_metrics]    


angles = np.linspace(0, 2*np.pi, len(plot_metrics_values), endpoint=False)
stats = np.concatenate((plot_metrics_values, [plot_metrics_values[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)
ax.set_rmax(1)

ax.tick_params(rotation='auto', pad = 5)
ax.set_thetagrids(angles[:-1] * 180/np.pi, plot_metrics)

ax.set_title("Metrics Summary", va='bottom')

plt.tight_layout()
plt.savefig(f'{visual_save_path}/metric_summary.png')
print(f'Save to {visual_save_path}/metric_summary.png')