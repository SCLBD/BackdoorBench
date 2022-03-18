# BDZOO2: backdoor defense and attack platform

- Datasets: `mnist, cifar10, cifar100, gtsrb, celeba, tiny, imagenet`
- Models: `resnet18, preactresnet18, resnet34, alexnet, vgg16, vgg19, squeezenet1_0, densenet161, inception_v3, googlenet, shufflenet_v2_x1_0, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet1_0`
- Target Types: `'all2one', 'all2all', 'cleanLabel'` (different attack varys)
- Attacks:
    - BadNets
      - paper: Gu, Tianyu, et al. “BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.” ArXiv:1708.06733 [Cs], Mar. 2019. arXiv.org, http://arxiv.org/abs/1708.06733.
    - Blended
      - Chen, Xinyun, et al. Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. Dec. 2017. arxiv.org, https://arxiv.org/abs/1712.05526v1.
    - SIG
      - Barni, M., Kallas, K., Tondi, B.: A new backdoor attack in cnns by training set corruption
  without label poisoning. In: IEEE International Conference on Image Processing (ICIP). pp.
  101–105. IEEE (2019)
    - SSBA
      - Li, Yuezun, et al. “Invisible Backdoor Attack with Sample-Specific Triggers.” ArXiv:2012.03816 [Cs], Aug. 2021. arXiv.org, http://arxiv.org/abs/2012.03816.
    - WaNet
      - Nguyen, Anh, and Anh Tran. “WaNet -- Imperceptible Warping-Based Backdoor Attack.” ArXiv:2102.10369 [Cs], Mar. 2021. arXiv.org, http://arxiv.org/abs/2102.10369.
    - InputAware
      - Nguyen, A., and A. Tran. “Input-Aware Dynamic Backdoor Attack.” NeurIPS, 2020.
- Defense methods: 

## Usage

### attack 

To use different attack method, you need to specify both the attack method script and yaml config file.

If you want to change setting, either change the parameter saved in yaml config file directly or specifiy it after yaml_path like `python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml --pratio 0.001`

The detailed descriptions for each attack may be put into the `add_args` function in each script. 

Examples: (assume in `./attack` folder)

 - BadNets

    `python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml`

 - Blended

    `python basicAttack.py --yaml_path ../config/basicAttack/default_blended.yaml`
 
 - SIG

    `python sigAttack.py --yaml_path ../config/sigAttack/default.yaml`

 - SSBA

    `python basicAttack.py --yaml_path ../config/basicAttack/default_ssba.yaml`

    (For this method, first you need to follow https://github.com/tancik/StegaStamp to train models for generating the poisoned data. Then place the poisoned image array to `attack_train_replace_imgs_path` and
`attack_test_replace_imgs_path`. Due to file size limitation we may not provide it in this repo.)

 - WaNet

    `python wanetAttack.py --yaml_path ../config/wanetAttack/default.yaml`    

 - InputAware

    `python inputAwareAttack.py --yaml_path ../config/inputAwareAttack/default.yaml`    

### defense

## Dependence


## Data Preparation
Please download datasets to `./data`. For GTSRB and TinyImagenet, we provide srcipts.

## Detalied Structure
- attack : all attack should be put here separately
- defense : all defense should be put here separately 
- config : all config file in yaml (all attack and defense config should all be put here separately)
- data : data file 
- experiment : analysis script and the final main entry will be put here 
- models : models that do not in the torchvision
- record : all experiment generated files and logs
- utils : frequent-use functions and other tools
  - bd_attack_specific_dataset : all special implementation of dataset-level backdoor 
    that CANNOT handle by 
    - bd_groupwise_transform (backdoor depends on mutliple batch-wise feed samples) or 
    - bd_dataset (backdoor only depends on each sample)
  - aggregate_block : frequent-use blocks in script
  - bd_img_transform : basic perturbation on img
  - bd_label_transform : basic transform on label
  - bd_non_mandatory_preparation: all non-mandatory preparation process for attack process, eg. train auto-encoder, selection of attack patterns that are non-dependent on victim model or dataset 
  - bd_groupwise_transform : for special case, such that data poison must be carried out groupwise, eg. HiddenTriggerBackdoorAttacks
  - bd_trainer : the training process can replicate for attack (for re-use, eg. noise training)
  - dataset : script for loading the dataset
  - dataset_preprocess : script for preprocess transforms on dataset 
  - backdoor_generate_pindex.py : some function for generation of poison index 
  - bd_dataset.py : the wrapper of backdoored datasets 
  - trainer_cls.py : some basic functions for classification case
- resource : pre-trained model (eg. auto-encoder for attack), or other large file (other than data)

## Results
