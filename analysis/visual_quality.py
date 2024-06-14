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

#### SSIM ####
visual_poison_indicator = np.array(get_poison_indicator_from_bd_dataset(visual_dataset))
bd_idx = np.where(visual_poison_indicator == 1)[0]

from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure()
ssim_list = []
if visual_poison_indicator.sum() > 0:
    print(f'Number Poisoned samples: {visual_poison_indicator.sum()}')
    # random choose two poisoned samples
    start_idx = 0
    for i in range(bd_idx.shape[0]):
        bd_sample = denormalizer(visual_dataset[i][0]).unsqueeze(0)
        with temporary_all_clean(visual_dataset):
            clean_sample =  denormalizer(visual_dataset[i][0]).unsqueeze(0)
        ssim_list.append(ssim(bd_sample, clean_sample))        
print(f'Average SSIM: {np.mean(ssim_list)}')


####### FFIM #######
visual_poison_indicator = np.array(get_poison_indicator_from_bd_dataset(visual_dataset))
bd_idx = np.where(visual_poison_indicator == 1)[0]

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64, normalize = True)
if visual_poison_indicator.sum() > 0:
    print(f'Number Poisoned samples: {visual_poison_indicator.sum()}')
    # random choose two poisoned samples
    start_idx = 0
    for i in range(bd_idx.shape[0]):
        bd_sample = denormalizer(visual_dataset[i][0]).unsqueeze(0)
        with temporary_all_clean(visual_dataset):
            clean_sample =  denormalizer(visual_dataset[i][0]).unsqueeze(0)
        fid.update(clean_sample, real=True)
        fid.update(bd_sample, real=False)
    fid_value = fid.compute().numpy()        
print(f'FID: {fid_value}')


###### PSNR ######
from torchmetrics.image.psnr import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio()
psnr_list = []

if visual_poison_indicator.sum() > 0:
    print(f'Number Poisoned samples: {visual_poison_indicator.sum()}')
    # random choose two poisoned samples
    start_idx = 0
    for i in range(bd_idx.shape[0]):
        bd_sample = denormalizer(visual_dataset[i][0]).unsqueeze(0)
        with temporary_all_clean(visual_dataset):
            clean_sample =  denormalizer(visual_dataset[i][0]).unsqueeze(0)
        psnr_list.append(psnr(bd_sample, clean_sample))        
print(f'Average PSNR: {np.mean(psnr_list)}')

quality_metrics = {}
quality_metrics['SSIM'] = np.mean(ssim_list)
quality_metrics['PSNR'] = np.mean(psnr_list)
quality_metrics['FID'] = fid_value

quality_metrics_df = pd.DataFrame(quality_metrics, index=[0])
quality_metrics_df.to_csv(f'{visual_save_path}/quality_metrics.csv', index=False)

print(f'Save to {visual_save_path}/quality_metrics.csv')


# visualize quality metrics is disabled since PSNR and SSIM are not comparable

# ### Visualization
# plot_metrics = list(quality_metrics.keys())
# plot_metrics_values = list(quality_metrics.values())    


# angles = np.linspace(0, 2*np.pi, len(plot_metrics_values), endpoint=False)
# stats = np.concatenate((plot_metrics_values, [plot_metrics_values[0]]))
# angles = np.concatenate((angles, [angles[0]]))

# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(angles, stats, 'o-', linewidth=2)
# ax.fill(angles, stats, alpha=0.25)
# ax.set_rmax(1)
# ax.set_rmin(-1)

# ax.tick_params(rotation='auto', pad = 5)
# ax.set_thetagrids(angles[:-1] * 180/np.pi, plot_metrics)

# ax.set_title("Quality Summary", va='bottom')

# plt.tight_layout()
# plt.savefig(f'{visual_save_path}/quality_metrics.png')
# print(f'Save to {visual_save_path}/quality_metrics.png')
