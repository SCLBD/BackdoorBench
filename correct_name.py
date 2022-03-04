# this script is for correct name of model
# only correct the attack_result!!!

import os, sys
import torch
from tqdm import tqdm

target_folder = './record'

for record_exp_name in tqdm(os.listdir(target_folder)):
    if os.path.isdir(f"{target_folder}/{record_exp_name}"):
        for file_name in os.listdir(f"{target_folder}/{record_exp_name}"):
            if file_name == "attack_result.pt":
                file_load = torch.load(f"{target_folder}/{record_exp_name}/{file_name}")
                # save the old
                if file_load['model_name'] == 'resnet18':
                    if not os.path.exists(f"{target_folder}/{record_exp_name}/old_attack_result.pt"):
                        torch.save(file_load, f"{target_folder}/{record_exp_name}/old_attack_result.pt")
                        file_load['model_name'] = 'preactresnet18'
                        torch.save(file_load, f"{target_folder}/{record_exp_name}/attack_result.pt")
                        print(f"{target_folder}/{record_exp_name} replace done")
                        with open(f"{target_folder}/{record_exp_name}/WARNING.txt", 'w') as f:
                            f.write(
                            "This record has only change the notation of 'resnet18' to 'preactresnet18' \n\
in info.pickle and attack_result.pt,\n\
be careful with all log file and text!\n\
(actually we use preactresnet18 for this exp !)"
                                )
            if file_name == 'info.pickle':
                file_load = torch.load(f"{target_folder}/{record_exp_name}/{file_name}")
                # save the old
                if file_load['model'] == 'resnet18':
                    if not os.path.exists(f"{target_folder}/{record_exp_name}/old_info.pickle"):
                        torch.save(file_load, f"{target_folder}/{record_exp_name}/old_info.pickle")
                        file_load['model'] = 'preactresnet18'
                        torch.save(file_load, f"{target_folder}/{record_exp_name}/info.pickle")
