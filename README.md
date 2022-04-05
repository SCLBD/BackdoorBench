# BackdoorBench: a comprehensive benchmark of backdoor attack and defense methods

![Python 3.6](https://img.shields.io/badge/python-3.7-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-DodgerBlue.svg?style=plastic)
![opencv 4.5.4.60](https://img.shields.io/badge/opencv-4.5.4.60-DodgerBlue.svg?style=plastic)

<!---
## [Overview](#overview)

<a href="#top">[Back to top]</a>
-->

BackdoorBench is a comprehensive benchmark of backdoor learning, which studies the adversarial vulnerablity of deep learning models in the training stage. It aims to provide **easy implementations** of mainstream backdoor attack and defense methods. Currently, we support:

- **Methods**
  - 6 Backdoor attack methods: [BadNets](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j), [Blended](https://arxiv.org/abs/1712.05526v1), [SIG](https://ieeexplore.ieee.org/document/8802997), [SSBA](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf), [WaNet](https://openreview.net/pdf?id=eEn8KTtJOx), [InputAware](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf)
  - 6 Backdoor defense methods: FT, [Spectral Signatures](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf), [AC](http://ceur-ws.org/Vol-2301/paper_18.pdf), [Fine-pruning](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13), [ABL](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf), [NAD](https://openreview.net/pdf?id=9l0K4OM-oXE)
- **Datasets**: CIFAR-10, GTSRB, Tiny ImageNet 
<!--- `mnist, cifar10, cifar100, gtsrb, celeba, tiny, imagenet`
(MNIST, CIFAR10, CIFAR100 using the pytorch official implementation, download when it is first executed. (TinyImageNet use third-party implementation, and it will be download when first executed.) The download script for GTSRB is in `./sh`. For CelebA and ImageNet, you need to download by yourself and change the dataset path argument. ) -->
- **Models**: Resnet18, PreactResnet18, VGG19
<!--- `resnet18, preactresnet18, resnet34, alexnet, vgg16, vgg19, squeezenet1_0, densenet161, inception_v3, googlenet, shufflenet_v2_x1_0, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet1_0` -->

For detailed structure and implementation details, you can refer to [detailed_structure.md](./docs/detailed_structure.md).

<!--- Note that, instead of implementing each individual method separately, we try to unify the workflow of different methods, by extracting some shared modules. Consequently, it can not only ensure fair implementations of different methods, but also facilitate other researchers to quickly implement their new methhods. -->

We also provide a [**public leaderboard**](https://backdoorbench.github.io/index.html) of evaluating above backdoor attacks against above backdoor defense methods.

BackdoorBench will be continuously updated to track the lastest advances of backddor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. **You are welcome to contribute your backdoor methods to BackdoorBench.**

---
<font size=5><center><b> Table of Contents </b> </center></font>

<!-- * [Overview](#overview) -->

* [Requirements](#requirements)

* [Usage](#usage)

  * [Attack](#attack)
  
  * [Defense](#defense)

* [Supported attacks](#supported-attacks)

* [Supported defenses](#supported-defsense)

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








### [Requirements](#requirements)

<a href="#top">[Back to top]</a>

You can run the following script to configurate necessary environment


```
sh ./sh/install.sh
```

<!--
```
keras==2.7.0
opencv-python==4.5.4.60
pandas==1.3.5
Pillow==8.4.0
scikit-learn==1.0.2
scikit-image==0.18.3
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.1
tqdm
```
-->

### [Usage](#usage)

<!--- <a href="#top">[Back to top]</a> -->


#### [Attack](#attack)

<a href="#top">[Back to top]</a>

This is a demo script of running badnets attack on cifar-10
```
python ./attack/badnets_attack.py --yaml_path ../config/badnets_attack/default.yaml --dataset cifar10 --dataset_path ../data/cifar10 --save_folder_name badnet_0_1
```
After attack you will get a folder with all files saved in ./record/<folder name in record>, including attack_result.pt for attack model and backdoored data, which will be used by following defense methods.
If you want to change the attack methods, dataset, save folder location, you should specify both the attack method script in ../attack and the YAML config file to use different attack methods.
The detailed descriptions for each attack may be put into the `add_args` function in each script.

[//]: # "You should specify both the attack method script and the YAML config file to use different attack methods. The detailed descriptions for each attack may be put into the `add_args` function in each script."

[//]: # "If you want to change the setting, either change the parameter saved in the YAML config file directly or specify it after `--yaml_path`"

[//]: # "like `python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml --pratio 0.001`"


[//]: # " - BadNets "

[//]: # "```shell"

[//]: # "cd attack "

[//]: # "python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml"

[//]: # "```"



#### [Defense](#defense)

<a href="#top">[Back to top]</a>

This is a demo script of running ac defense on cifar-10 for badnet attack. Before defense you need to run badnet attack on cifar-10 at first. Then you use the folder name as result_file.

```
python ./defense/ac/ac.py --result_file badnet_0_1
```


If you want to change the defense methods and the setting for defense, you should specify both the attack method script in ../defense and the YAML config file to use different defense methods.
The detailed descriptions for each defense may be put into the `add_args` function in each script.

### [Supported attacks](#supported-attacks)

<a href="#top">[Back to top]</a>

|            | File name                                             | Paper                                                        |
| ---------- |-------------------------------------------------------| ------------------------------------------------------------ |
| BadNets    | [badnets_attack.py](./attack/badnets_attack.py)       | [BadNets: Identifying Vulnerabilities in  the Machine Learning Model Supply Chain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j) IEEE Access 2019 |
| Blended    | [blended_attack.py](./attack/blended_attack.py)       | [Targeted Backdoor Attacks on Deep  Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526v1) Arxiv 2017 |
| SIG        | [sig_attack.py](./attack/sig_attack.py)               | [A new backdoor attack in  cnns by training set corruption](https://ieeexplore.ieee.org/document/8802997) ICIP 2019 |
| SSBA       | [ssba_attack.py](./attack/ssba_attack.py)             | [Invisible Backdoor Attack with  Sample-Specific Triggers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf) ICCV 2021 |
| WaNet      | [wanet_attack.py](./attack/wanet_attack.py)            | [WaNet -- Imperceptible  Warping-Based Backdoor Attack](https://openreview.net/pdf?id=eEn8KTtJOx) ICLR 2021 |
| InputAware | [inputaware_attack.py](./attack/inputaware_attack.py) | [Input-Aware Dynamic Backdoor Attack](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf) NeurIPS 2020 |


### [Supported defenses](#supported-defsense) 

<a href="#top">[Back to top]</a>

|       | File name                 | Paper                |
| :------------- |:-------------|:-----|
| FT| [finetune.py](./defense/ft/finetune.py) | standard fine-tuning|
| Spectral Signatures| [spectral_signature.py](./defense/spectral_signatural/spectral_signature.py)    | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf) NeurIPS 2018 |
| AC| [ac.py](./defense/ac/ac.py)       | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](http://ceur-ws.org/Vol-2301/paper_18.pdf) ceur-ws 2018 |
| Fine-pruning| [fineprune.py](./defense/fp/fineprune.py)    | [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) RAID 2018 |
| ABL| [abl.py](./defense/abl/abl.py)    | [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf) NeurIPS 2021|
| NAD| [nad.py](./defense/nad/nad.py)   | [Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks](https://openreview.net/pdf?id=9l0K4OM-oXE) ICLR 2021 |


### [Results](#results)

<a href="#top">[Back to top]</a>

We present partial results on cifar10 with poison ratio = 10% here. For complete results, please refer to our [leaderboard](https://backdoorbench.github.io/leaderboard-cifar10.html).

|                |  Defenses →    | No Defense | No Defense | No Defense | AC     | AC     | AC     | Fine_pruning | Fine_pruning | Fine_pruning | Fine_tuning | Fine_tuning | Fine_tuning | Abl    | Abl     | Abl    | NAD    | NAD    | NAD    | Spectral_signatural | Spectral_signatural | Spectral_signatural |
| -------------- | ------------ | ---------- |---------| ---------- | ------ |--------| ------ | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ------ | ------- | ------ | ------ | ------ | ------ | ------------------- | ------------------- | ------------------- |
|  **Models ↓**   |  **Attacks ↓**   | CA         | ASR     | RA         | CA     | ASR    | RA     | CA           | ASR          | RA           | CA          | ASR         | RA          | CA     | ASR     | RA     | CA     | ASR    | RA     | CA                  | ASR                 | RA                  |
| PreactResnet18 | Badnet       | 91.49%     | 96.30%  | 67.09%     | 84.50% | 18.30% | 73.89% | 90.98%       | 6.32%        | 87.29%       | 88.81%      | 2.56%       | 88.19%      | 84.32% | 72.23%  | 79.18% | 64.96% | 3.69%  | 59.86% | 89.42%              | 93.43%              | 80.02%              |
| PreactResnet18 | Blended      | 93.16%     | 99.79%  | 58.81%     | 87.56% | 28.52% | 51.43% | 93.16%       | 99.79%       | 58.81%       | 91.10%      | 15.26%      | 63.38%      | 82.47% | 24.09%  | 64.52% | 63.95% | 3.06%  | 51.76% | 91.19%              | 99.41%              | 56.56%              |
| PreactResnet18 | SIG          | 84.64%     | 98.24%  | 82.44%     | 80.93% | 3.04%  | 78.41% | 53.48%       | 3.34%        | 11.73%       | 89.42%      | 8.54%       | 76.84%      | 72.96% | 32.84%  | 78.97% | 64.40% | 0.96%  | 47.28% | 82.73%              | 97.01%              | 79.91%              |
| PreactResnet18 | SSBA         | 93.37%     | 98.92%  | 7.76%      | 83.40% | 26.00% | 8.19%  | NAN          | NAN          | NAN          | 89.84%      | 14.63%      | 9.29%       | 75.82% | 97.61%  | 9.07%  | 64.11% | 11.09% | 9.48%  | 91.29%              | 98.79%              | 8.41%               |
| PreactResnet18 | Wanet        | 91.75%     | 83.18%  | 19.98%     | 85.58% | 19.77% | 42.28% | 90.99%       | 16.14%       | 16.22%       | 92.60%      | 11.89%      | 23.20%      | 79.28% | 84.10%  | 45.11% | 63.58% | 60.49% | 15.02% | 90.93%              | 75.24%              | 46.71%              |
| PreactResnet18 | Input  Aware | 91.72%     | 84.49%  | 18.23%     | 87.53% | 19.01% | 32.00% | 91.25%       | 4.43%        | 12.88%       | 91.49%      | 11.20%      | 22.24%      | 83.37% | 84.69%  | 33.97% | 63.71% | 42.86% | 18.00% | 91.27%              | 85.41%              | 36.14%              |
| Resnet18       | Badnet       | 83.73%     | 90.51%  | 68.71%     | 75.58% | 15.08% | 68.79% | 11.30%       | 0.12%        | 11.11%       | 80.34%      | 4.01%       | 77.93%      | 74.86% | 14.90%  | 75.99% | 50.24% | 5.83%  | 47.52% | 82.24%              | 86.80%              | 71.43%              |
| Resnet18       | Blended      | 84.32%     | 98.72%  | 62.20%     | 78.25% | 18.83% | 59.96% | 65.44%       | 11.24%       | 28.38%       | 80.11%      | 7.09%       | 68.04%      | 73.76% | 43.12%  | 68.29% | 48.98% | 7.80%  | 45.69% | 81.62%              | 97.71%              | 61.03%              |
| Resnet18       | SIG          | 77.42%     | 97.54%  | 76.04%     | 72.09% | 2.80%  | 73.64% | 57.97%       | 0.12%        | 37.51%       | 79.70%      | 6.60%       | 71.77%      | 61.47% | 98.04%  | 73.17% | 47.40% | 3.80%  | 44.10% | 75.25%              | 96.78%              | 74.59%              |
| Resnet18       | SSBA         | 84.52%     | 88.63%  | 9.02%      | 75.67% | 20.09% | 8.96%  | 29.32%       | 5.26%        | 10.33%       | 79.50%      | 10.13%      | 10.08%      | 75.39% | 61.38%  | 9.56%  | 47.71% | 10.42% | 10.09% | 81.49%              | 89.54%              | 9.02%               |
| Resnet18       | Wanet        | 83.73%     | 59.34%  | 20.26%     | 76.47% | 17.44% | 35.61% | 22.21%       | 0.53%        | 11.52%       | 84.00%      | 10.11%      | 21.88%      | 65.24% | 99.98%  | 30.97% | 47.72% | 23.43% | 14.83% | 82.57%              | 58.28%              | 38.62%              |
| Resnet18       | Input  Aware | 83.25%     | 81.03%  | 19.66%     | 75.65% | 17.41% | 28.93% | 66.35%       | 8.77%        | 12.19%       | 81.45%      | 8.92%       | 22.13%      | 65.99% | 91.80%  | 22.30% | 47.79% | 15.69% | 12.96% | 82.06%              | 81.50%              | 28.49%              |
| VGG19          | Badnet       | 88.99%     | 96.22%  | 3.69%      | 86.46% | 92.33% | 7.09%  | 15.88%       | 11.48%       | 10.92%       | 88.18%      | 34.92%      | 62.17%      | 85.00% | 88.69%  | 9.70%  | 54.44% | 5.38%  | 53.07% | 87.09%              | 93.86%              | 5.63%               |
| VGG19          | Blended      | 90.05%     | 99.27%  | 0.66%      | 86.79% | 98.24% | 1.56%  | 25.80%       | 0.00%        | 12.70%       | 88.81%      | 95.64%      | 4.13%       | 59.11% | 93.11%  | 3.40%  | 51.55% | 5.60%  | 45.93% | 88.42%              | 99.33%              | 0.63%               |
| VGG19          | SIG          | 81.92%     | 99.62%  | 0.30%      | 79.49% | 99.66% | 0.27%  | 44.31%       | 0.00%        | 13.80%       | 86.99%      | 90.67%      | 6.93%       | 10.00% | 0.00%   | 9.44%  | 53.75% | 4.28%  | 26.66% | 80.82%              | 99.83%              | 0.13%               |
| VGG19          | SSBA         | 89.65%     | 96.22%  | 0.46%      | 86.39% | 93.73% | 0.81%  | 60.03%       | 16.01%       | 9.06%        | 88.59%      | 77.23%      | 2.51%       | 29.49% | 28.97%  | 7.88%  | 51.88% | 11.64% | 9.30%  | 88.31%              | 95.23%              | 0.61%               |
| VGG19          | Wanet        | 90.43%     | 78.18%  | 20.30%     | 88.18% | 76.24% | 12.37% | 86.07%       | 3.90%        | 21.41%       | 90.63%      | 21.43%      | 20.54%      | 10.00% | 100.00% | 0.00%  | 51.96% | 18.52% | 16.00% | 88.84%              | 88.00%              | 6.76%               |
| VGG19          | Input  Aware | 89.52%     | 88.09%  | 12.29%     | 87.08% | 77.14% | 11.47% | 82.06%       | 0.49%        | 16.53%       | 90.31%      | 25.59%      | 14.02%      | 76.43% | 60.11%  | 17.89% | 40.19% | 10.26% | 13.88% | 88.43%              | 79.11%              | 11.28%              |

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

This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) and [Shenzhen Research Institute of Big Data](http://www.sribd.cn/en) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
