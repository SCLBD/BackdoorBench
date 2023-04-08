This is for ImageNet only. (Still under construction)

Because we only considered backdoor attacks on small and medium datasets at the beginning of the design, we do not have better support for ImageNet datasets. You will likely fail your training process when you use ImageNet as your target dataset due to insufficient RAM. So this folder is dedicated to backdoor attacks on ImageNet.

1. Download ImageNet data by yourself and put in `data` folder.
2. Use the script to generate the data for training and validation.
eg. `multi_generate_poison_badnet.py` and `generate_poison_val_badnet.py` for BadNets.
3. Run `train.py` with proper setting specified.

Example Results (With PreAct-ResNet18, 0.1% poison rate. For all other detailed settings, you can refer to corresponding scripts. ) 

|         | ACC      | ASR      | RA       |
| ------- | -------- | -------- | -------- |
| BadNets | 69.20923 | 75.86055 | 0.338413 |
| Blended | 69.23923 | 98.58628 | 0.110134 |