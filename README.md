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
  - 6 Backdoor defense methods: FT, [Spectral Signatures](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf), [AC](http://ceur-ws.org/Vol-2301/paper_18.pdf), [Fine-pruning](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13), [ABL](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf), [NAD](https://openreview.net/pdf?id=9l0K4OM-oXE), [NC](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), [DBD](https://arxiv.org/pdf/2202.03423.pdf), [ANP](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)
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
python -u ./attack/badnets_attack.py --yaml_path ../config/BadNetsAttack/default.yaml --dataset cifar10 --dataset_path ../data/cifar10 --save_folder_name badnet_0_1
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
| NC | [nc.py](./defense/nc/nc.py) | [Neural Cleanse: Identifying And Mitigating Backdoor Attacks In Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), IEEE S&P 2019 |
| DBD | [dbd.py](./defense/dbd/dbd.py) | [Backdoor Defense Via Decoupling The Training Process](https://arxiv.org/pdf/2202.03423.pdf) ICLR 2022 |
| ANP | [anp.py](./defense/anp/anp.py) | [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf) NeurIPS 2021 |


### [Results](#results)

<a href="#top">[Back to top]</a>

We present partial results on cifar10 with poison ratio = 10% here. For complete results, please refer to our [leaderboard](https://backdoorbench.github.io/leaderboard-cifar10.html).

|                |      Defenses →        | No Defense | No Defense | No Defense | AC     | AC     | AC     | Fine_pruning | Fine_pruning | Fine_pruning | Fine_tuning | Fine_tuning | Fine_tuning | Abl    | Abl     | Abl    | NAD    | NAD    | NAD    | Spectral_signatural | Spectral_signatural | Spectral_signatural |
| -------------- | ------------ | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ------ | ------- | ------ | ------ | ------ | ------ | ------------------- | ------------------- | ------------------- |
| **Models ↓**   |  **Attacks ↓**  | CA         | ASR        | RC         | CA     | ASR    | RC     | CA           | ASR          | RC           | CA          | ASR         | RC          | CA     | ASR     | RC     | CA     | ASR    | RC     | CA                  | ASR                 | RC                  |
| PreactResnet18 | Badnet       | 91.96%     | 95.23%     | 69.18%     | 83.52% | 17.49% | 75.42% | 92.14%       | 76.42%       | 22.99%       | 88.34%      | 2.90%       | 87.72%      | 16.91% | 0.00%   | 18.23% | 63.26% | 4.94%  | 62.73% | 88.79%              | 95.27%              | 4.39%               |
| PreactResnet18 | Blended      | 93.60%     | 99.88%     | 57.32%     | 88.08% | 28.28% | 55.90% | 93.48%       | 99.02%       | 0.94%        | 90.96%      | 12.84%      | 67.62%      | 10.02% | 98.32%  | 0.40%  | 63.26% | 3.71%  | 54.29% | 91.58%              | 99.10%              | 0.87%               |
| PreactResnet18 | SIG          | 84.56%     | 97.80%     | 81.94%     | 80.23% | 3.52%  | 78.46% | 19.50%       | 0.00%        | 12.90%       | 89.38%      | 8.56%       | 77.59%      | 9.99%  | 0.00%   | 11.11% | 63.26% | 2.27%  | 30.80% | 82.73%              | 98.04%              | 1.91%               |
| PreactResnet18 | SSBA         | 93.28%     | 98.77%     | 7.92%      | 83.64% | 28.83% | 7.89%  | 93.04%       | 97.27%       | 0.33%        | 89.32%      | 17.47%      | 9.20%       | 11.24% | 0.00%   | 11.20% | 63.26% | 12.49% | 9.68%  | 89.31%              | 98.90%              | 0.06%               |
| PreactResnet18 | Wanet        | 91.35%     | 88.25%     | 19.69%     | 84.67% | 20.66% | 42.19% | 91.32%       | 29.64%       | 13.59%       | 93.08%      | 11.48%      | 22.91%      | 9.63%  | 99.98%  | 0.00%  | 63.26% | 0.21%  | 11.98% | 90.26%              | 75.46%              | 13.22%              |
| PreactResnet18 | Input  Aware | 90.74%     | 84.18%     | 18.97%     | 87.51% | 17.46% | 35.51% | 90.75%       | 12.48%       | 12.32%       | 91.37%      | 10.91%      | 21.71%      | 9.42%  | 7.68%   | 10.83% | 63.26% | 0.23%  | 12.31% | 90.43%              | 86.53%              | 9.17%               |
| Resnet18       | Badnet       | 83.76%     | 88.71%     | 71.64%     | 74.92% | 17.04% | 67.38% | 24.67%       | 0.32%        | 16.80%       | 79.88%      | 4.50%       | 77.69%      | 24.32% | 0.00%   | 25.88% | 48.24% | 6.87%  | 46.97% | 80.54%              | 89.32%              | 8.90%               |
| Resnet18       | Blended      | 84.64%     | 98.48%     | 62.67%     | 77.48% | 18.03% | 60.61% | 25.02%       | 0.00%        | 18.73%       | 80.63%      | 6.24%       | 69.22%      | 15.05% | 0.00%   | 21.50% | 48.24% | 9.20%  | 42.81% | 82.07%              | 96.53%              | 2.82%               |
| Resnet18       | SIG          | 77.25%     | 97.10%     | 77.94%     | 72.07% | 2.28%  | 73.11% | 41.24%       | 0.11%        | 30.57%       | 79.31%      | 7.21%       | 72.28%      | 20.65% | 0.00%   | 16.72% | 48.24% | 4.41%  | 39.52% | 75.44%              | 84.97%              | 11.91%              |
| Resnet18       | SSBA         | 84.10%     | 89.47%     | 8.91%      | 76.37% | 19.82% | 8.83%  | 29.16%       | 5.16%        | 10.33%       | 79.62%      | 11.02%      | 10.08%      | 24.55% | 0.00%   | 11.13% | 48.24% | 12.39% | 9.56%  | 80.84%              | 86.69%              | 1.37%               |
| Resnet18       | Wanet        | 81.45%     | 73.30%     | 22.50%     | 77.48% | 15.62% | 34.97% | 29.84%       | 2.83%        | 12.63%       | 83.63%      | 11.62%      | 23.64%      | 10.00% | 100.00% | 0.00%  | 48.24% | 5.76%  | 11.94% | 83.07%              | 66.06%              | 18.71%              |
| Resnet18       | Input  Aware | 80.91%     | 87.75%     | 20.90%     | 75.41% | 17.04% | 27.23% | 60.47%       | 3.13%        | 11.02%       | 80.80%      | 9.87%       | 23.41%      | 10.91% | 99.99%  | 0.01%  | 48.24% | 2.90%  | 12.50% | 83.07%              | 80.07%              | 10.24%              |
| VGG19          | Badnet       | 89.42%     | 95.87%     | 3.81%      | 86.46% | 94.96% | 4.56%  | 13.90%       | 0.51%        | 11.11%       | 88.09%      | 59.23%      | 39.10%      | 10.00% | 100.00% | 0.00%  | 53.75% | 5.48%  | 53.47% | 87.47%              | 94.37%              | 5.19%               |
| VGG19          | Blended      | 90.08%     | 99.23%     | 0.71%      | 87.51% | 99.19% | 0.81%  | 18.32%       | 0.00%        | 11.47%       | 88.63%      | 95.20%      | 4.54%       | 10.00% | 100.00% | 0.00%  | 52.75% | 6.24%  | 48.98% | 88.41%              | 99.20%              | 0.72%               |
| VGG19          | SIG          | 81.84%     | 99.73%     | 0.26%      | 79.95% | 99.69% | 0.24%  | 29.79%       | 0.00%        | 11.33%       | 86.68%      | 38.41%      | 34.16%      | 10.00% | 100.00% | 0.00%  | 53.09% | 1.18%  | 32.37% | 80.51%              | 98.68%              | 1.11%               |
| VGG19          | SSBA         | 89.66%     | 96.56%     | 0.33%      | 86.72% | 96.46% | 0.33%  | 59.35%       | 0.00%        | 10.93%       | 88.13%      | 78.77%      | 2.22%       | 10.00% | 100.00% | 0.00%  | 52.89% | 10.90% | 9.56%  | 88.24%              | 96.14%              | 0.46%               |
| VGG19          | Wanet        | 89.09%     | 81.91%     | 14.26%     | 88.37% | 85.49% | 7.50%  | 85.60%       | 47.67%       | 13.77%       | 91.06%      | 31.74%      | 16.58%      | 10.00% | 100.00% | 0.00%  | 52.93% | 4.92%  | 17.18% | 89.37%              | 81.10%              | 10.36%              |
| VGG19          | Input  Aware | 86.76%     | 86.60%     | 12.47%     | 87.47% | 64.70% | 17.11% | 82.82%       | 4.48%        | 16.96%       | 90.14%      | 40.17%      | 11.96%      | 10.00% | 100.00% | 0.00%  | 53.12% | 1.36%  | 14.56% | 88.94%              | 74.91%              | 13.16%              |



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


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) and [Shenzhen Research Institute of Big Data](http://www.sribd.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
