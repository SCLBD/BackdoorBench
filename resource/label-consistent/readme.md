This folder contains the script to generate the poison data for label-consistent attack. 

You can replace PGD with other adversarial attack module by yourself (Setting is also written in craft_adv_dataset.py). 

command:
```
python craft_adv_dataset.py --dataset cifar10 
python craft_adv_dataset.py --dataset cifar100 
python craft_adv_dataset.py --dataset tiny 
python craft_adv_dataset.py --dataset gtsrb 
```