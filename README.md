# BackdoorBench: a comprehensive benchmark of backdoor attack and defense methods

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.6.0-DodgerBlue.svg?style=plastic)

<!---
## [Overview](#overview)

<a href="#top">[Back to top]</a>
-->

BackdoorBench is a comprehensive benchmark of backdoor learning. It aims to provide **easy implementations** of mainstream backdoor attack and defense methods, as well as a [**public leaderboard**](https://backdoorbench.github.io/index.html) of evaluating existing backdoor attack and defense methods. Currently, we support:

- **Methods**
  - 6 Backdoor attack methods: [BadNets](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j), [Blended](https://arxiv.org/abs/1712.05526v1), [SIG](https://ieeexplore.ieee.org/document/8802997), [SSBA](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf), [WaNet](https://openreview.net/pdf?id=eEn8KTtJOx), [InputAware](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf)
  - 6 Backdoor defense methods: 
- **Datasets**: CIFAR-10, GTSRB, Tiny ImageNet 
<!--- `mnist, cifar10, cifar100, gtsrb, celeba, tiny, imagenet`
(MNIST, CIFAR10, CIFAR100 using the pytorch official implementation, download when it is first executed. (TinyImageNet use third-party implementation, and it will be download when first executed.) The download script for GTSRB is in `./sh`. For CelebA and ImageNet, you need to download by yourself and change the dataset path argument. ) -->
- **Models**: Resnet18, PreactResnet18, VGG19
<!--- `resnet18, preactresnet18, resnet34, alexnet, vgg16, vgg19, squeezenet1_0, densenet161, inception_v3, googlenet, shufflenet_v2_x1_0, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet1_0` -->


<!--- Note that, instead of implementing each individual method separately, we try to unify the workflow of different methods, by extracting some shared modules. Consequently, it can not only ensure fair implementations of different methods, but also facilitate other researchers to quickly implement their new methhods. -->

BackdoorBench will be continuously updated to track the lastest advances of backddor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. **You are welcome to contribute your backdoor methods to BackdoorBench.**

---
<font size=5><center><b> Table of Contents </b> </center></font>

* [Overview](#overview)

* [Usage](#usage)

  * [attack](#attack)
  
  * [defense](#defense)
  
* [Dependence](#dependence)

* [Data Preparation](#data_preparation)

* [Detalied Structure](#detailed_structure)

* [Results](#results)

* [Citation](#citation)

* [Copyright](#copyright)

---

<!---
### Datasets: 
`mnist, cifar10, cifar100, gtsrb, celeba, tiny, imagenet`
(MNIST, CIFAR10, CIFAR100 using the pytorch official implementation, download when it is first executed.(TinyImageNet use third-party implementation, and it will be download when first executed.) The download script for GTSRB is in `./sh`. For CelebA and ImageNet, you need to download by yourself and change the dataset path argument. )
### Models: 
`resnet18, preactresnet18, resnet34, alexnet, vgg16, vgg19, squeezenet1_0, densenet161, inception_v3, googlenet, shufflenet_v2_x1_0, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet1_0`
-->

[//]: # "### Target Types: `'all2one', 'all2all', 'cleanLabel'` &#40;different attack varys&#41;"
### Attacks:
|            | File name                                           | Paper                                                        |
| ---------- | --------------------------------------------------- | ------------------------------------------------------------ |
| BadNets    | [badnetsattack.py](./attack/badnetsattack.py)       | [BadNets: Identifying Vulnerabilities in  the Machine Learning Model Supply Chain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j) IEEE Access 2019 |
| Blended    | [blendedattack.py](./attack/blendedattack.py)       | [Targeted Backdoor Attacks on Deep  Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526v1) Arxiv 2017 |
| SIG        | [sigattack.py](./attack/sigattack.py)               | [A new backdoor attack in  cnns by training set corruption](https://ieeexplore.ieee.org/document/8802997) ICIP 2019 |
| SSBA       | [ssbaattack.py](./attack/ssbaattack.py)             | [Invisible Backdoor Attack with  Sample-Specific Triggers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf) ICCV 2021 |
| WaNet      | [wanetattack.py](./attack/wanetattack.py)           | [WaNet -- Imperceptible  Warping-Based Backdoor Attack](https://openreview.net/pdf?id=eEn8KTtJOx) ICLR 2021 |
| InputAware | [inputawareattack.py](./attack/inputawareattack.py) | [Input-Aware Dynamic Backdoor Attack](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf) NeurIPS 2020 |

[//]: #
[//]: # "- BadNets"

[//]: # "  - Gu, Tianyu, et al. “BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.” ArXiv:1708.06733 [Cs], Mar. 2019. arXiv.org, http://arxiv.org/abs/1708.06733."

[//]: # "- Blended"

[//]: # "  - Chen, Xinyun, et al. Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. Dec. 2017. arxiv.org, https://arxiv.org/abs/1712.05526v1."

[//]: # "- SIG"

[//]: # "  - Barni, M., Kallas, K., Tondi, B.: A new backdoor attack in cnns by training set corruption"

[//]: # "  without label poisoning. In: IEEE International Conference on Image Processing &#40;ICIP&#41;. pp."

[//]: # "  101–105. IEEE &#40;2019&#41;"

[//]: # "- SSBA"

[//]: # "  - Li, Yuezun, et al. “Invisible Backdoor Attack with Sample-Specific Triggers.” ArXiv:2012.03816 [Cs], Aug. 2021. arXiv.org, http://arxiv.org/abs/2012.03816."

[//]: # "- WaNet"

[//]: # "  - Nguyen, Anh, and Anh Tran. “WaNet -- Imperceptible Warping-Based Backdoor Attack.” ArXiv:2102.10369 [Cs], Mar. 2021. arXiv.org, http://arxiv.org/abs/2102.10369."

[//]: # "- InputAware"

[//]: # "  - Nguyen, A., and A. Tran. “Input-Aware Dynamic Backdoor Attack.” NeurIPS, 2020."
### Defense :

|       | File name                 | Paper                |
| :------------- |:-------------|:-----|
| FT| [finetune.py](./defense/ft/finetune.py) | standard fine-tuning|
| Spectral Signatures| [spectral_signatural.py](./defense/spectral_signatural/spectral_signatural.py)      | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf) NeurIPS 2018 |
| AC| [ac.py](./defense/ac/ac.py)       | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](http://ceur-ws.org/Vol-2301/paper_18.pdf) ceur-ws 2018 |
| Fine-pruning| [fineprune.py](./defense/fp/fineprune.py)    | [Fine-pruning: Defending againstbackdooring attacks on deep neural networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) International Symposium on Research in Attacks, Intrusions, and Defenses(2018) |
| ABL| [abl.py](./defense/abl/abl.py)    | [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf) NeurIPS 2021|
| NAD| [nad.py](./defense/nad/nad.py)   | [NEURAL ATTENTION DISTILLATION: ERASING BACKDOOR TRIGGERS FROM DEEP NEURAL NETWORKS](https://openreview.net/pdf?id=9l0K4OM-oXE) ICLR 2021|






### Detailed Structure and Implementation Details
You can refer to `./docs` folder.

## [Dependence](#dependence)

<a href="#top">[Back to top]</a>

```
keras==2.7.0
opencv-python==4.5.4.60
pandas==1.3.5
Pillow==8.4.0
scikit-image==0.18.3
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.1
tqdm
```

## [Usage](#usage)

<a href="#top">[Back to top]</a>


### [attack](#attack)

<a href="#top">[Back to top]</a>


You should specify both the attack method script and the YAML config file to use different attack methods. The detailed descriptions for each attack may be put into the `add_args` function in each script.

[//]: # (If you want to change the setting, either change the parameter saved in the YAML config file directly or specify it after `--yaml_path`)

[//]: # "like `python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml --pratio 0.001`"


 - BadNets 
```shell
cd attack 
python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml
```

[//]: # " - Blended"

[//]: #
[//]: # "    `python basicAttack.py --yaml_path ../config/basicAttack/default_blended.yaml`"

[//]: # " "
[//]: # " - SIG"

[//]: #
[//]: # "    `python sigAttack.py --yaml_path ../config/sigAttack/default.yaml`"

[//]: #
[//]: # " - SSBA"

[//]: #
[//]: # "    `python basicAttack.py --yaml_path ../config/basicAttack/default_ssba.yaml`"

[//]: # (&#40;For SSBA, first, you need to follow https://github.com/tancik/StegaStamp to train models for generating the poisoned data. Then place the poisoned image array to `attack_train_replace_imgs_path` and)

[//]: # (`attack_test_replace_imgs_path`. Due to file size limitations, we may not provide it in this repo.&#41;)

[//]: # " - WaNet"

[//]: #
[//]: # "    `python wanetAttack.py --yaml_path ../config/wanetAttack/default.yaml`    "

[//]: #
[//]: # " - InputAware"

[//]: #
[//]: # "    `python inputAwareAttack.py --yaml_path ../config/inputAwareAttack/default.yaml`    "

### [defense](#defense)

<a href="#top">[Back to top]</a>

You should specify both **the defense method script** and **the attack result** to use different attack methods. The yaml config is in defense method. 

Examples: 

- AC
```shell
python -u ./defense/ac/ac.py --result_file badnet_0_1
```







[//]: # "# [Data Preparation]&#40;#data_preparation&#41;"

[//]: #
[//]: # "<a href="#top">[Back to top]</a>"

[//]: #
[//]: # "Please download datasets to `./data`. For GTSRB and TinyImagenet, we provide srcipts."

[//]: # "# [Detalied Structure]&#40;#detailed_structure&#41;"

[//]: #
[//]: # "<a href="#top">[Back to top]</a>"

[//]: #
[//]: # "- attack : all attack should be put here separately"

[//]: # "- defense : all defense should be put here separately "

[//]: # "- config : all config file in yaml &#40;all attack and defense config should all be put here separately&#41;"

[//]: # "- data : data file "

[//]: # "- experiment : analysis script and the final main entry will be put here "

[//]: # "- models : models that do not in the torchvision"

[//]: # "- record : all experiment generated files and logs"

[//]: # "- utils : frequent-use functions and other tools"

[//]: # "  - bd_attack_specific_dataset : all special implementation of dataset-level backdoor "

[//]: # "    that CANNOT handle by "

[//]: # "    - bd_groupwise_transform &#40;backdoor depends on mutliple batch-wise feed samples&#41; or "

[//]: # "    - bd_dataset &#40;backdoor only depends on each sample&#41;"

[//]: # "  - aggregate_block : frequent-use blocks in script"

[//]: # "  - bd_img_transform : basic perturbation on img"

[//]: # "  - bd_label_transform : basic transform on label"

[//]: # "  - bd_non_mandatory_preparation: all non-mandatory preparation process for attack process, eg. train auto-encoder, selection of attack patterns that are non-dependent on victim model or dataset "

[//]: # "  - bd_groupwise_transform : for special case, such that data poison must be carried out groupwise, eg. HiddenTriggerBackdoorAttacks"

[//]: # "  - bd_trainer : the training process can replicate for attack &#40;for re-use, eg. noise training&#41;"

[//]: # "  - dataset : script for loading the dataset"

[//]: # "  - dataset_preprocess : script for preprocess transforms on dataset "

[//]: # "  - backdoor_generate_pindex.py : some function for generation of poison index "

[//]: # "  - bd_dataset.py : the wrapper of backdoored datasets "

[//]: # "  - trainer_cls.py : some basic functions for classification case"

[//]: # "- resource : pre-trained model &#40;eg. auto-encoder for attack&#41;, or other large file &#40;other than data&#41;"

## [Results](#results)

<a href="#top">[Back to top]</a>


## [Citation](#citation)

<a href="#top">[Back to top]</a>

If interested, you can read our recent works about backdoor learning, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@inproceedings{dbd-backdoor-defense-iclr2022,
title={Backdoor Defense via Decoupling the Training Process},
author={Huang, Kunzhe and Li, Yiming and Wu, Baoyuan and Qin, Zhan and Ren, Kui},
booktitle={International Conference on Learning Representations},
year={2022}
}

@inproceedings{ssba-backdoor-attack-iccv2021,
title={Invisible backdoor attack with sample-specific triggers},
author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={16463--16472},
year={2021}
}
```


## [Copyright](#copyright)

<a href="#top">[Back to top]</a>

<!-- This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) and [Shenzhen Research Institute of Big Data](http://www.sribd.cn/en) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.  -->

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
