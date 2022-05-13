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


### [Results](#results)

<a href="#top">[Back to top]</a>

We present partial results on cifar10 with poison ratio = 10% here. For complete results, please refer to our [leaderboard](https://backdoorbench.github.io/leaderboard-cifar10.html).

|                    | Backdoor Defense → | No Defense | No Defense | No Defense | AC     | AC      | AC     | Fine Pruning | Fine Pruning | Fine Pruning | Fine Tuning | Fine Tuning | Fine Tuning | ABL    | ABL     | ABL    | NAD    | NAD    | NAD    | Spectral Signatural | Spectral Signatural | Spectral Signatural | DBD    | DBD    | DBD    | NC    | NC     | NC    | ANP    | ANP     | ANP    |
| ------------------ | ------------------ | ---------- | ---------- | ---------- | ------ | ------- | ------ | ------------ | ------------ | ------------ | ----------- | ----------- | ----------- | ------ | ------- | ------ | ------ | ------ | ------ | ------------------- | ------------------- | ------------------- | ------ | ------ | ------ | ----- | ------ | ----- | ------ | ------- | ------ |
| Targeted Model     | Backdoor Attack↓   | CA(%)      | ASR(%)     | RA(%)      | CA(%)  | ASR(%)  | RA(%)  | CA(%)        | ASR(%)       | RA(%)        | CA(%)       | ASR(%)      | RA(%)       | CA(%)  | ASR(%)  | RA(%)  | CA(%)  | ASR(%) | RA(%)  | CA(%)               | ASR(%)              | RA(%)               | CA(%)  | ASR(%) | RA(%)  | CA(%) | ASR(%) | RA(%) | CA(%)  | ASR(%)  | RA(%)  |
| preactresnet18     | badnet             | 91.32%     | 95.03%     | 4.64%      | 88.80% | 86.23%  | 13.28% | 91.08%       | 76.38%       | 22.93%       | 90.48%      | 1.60%       | 89.87%      | 14.64% | 0.00%   | 16.17% | 63.49% | 5.21%  | 63.22% | 89.98%              | 92.41%              | 7.20%               | 75.75% | 8.27%  | 73.60% | NA    | NA     | NA    | 41.16% | 0.00%   | 44.81% |
| preactresnet18     | blended            | 93.47%     | 99.92%     | 0.08%      | 88.52% | 99.72%  | 0.28%  | 93.22%       | 99.28%       | 0.70%        | 92.70%      | 96.28%      | 3.43%       | 11.28% | 0.00%   | 13.41% | 63.49% | 3.56%  | 53.19% | 90.35%              | 99.84%              | 0.14%               | 53.52% | 99.84% | 0.07%  | NA    | NA     | NA    | 81.96% | 0.11%   | 56.56% |
| preactresnet18     | sig                | 84.48%     | 98.27%     | 1.72%      | 82.41% | 94.61%  | 5.17%  | 10.16%       | 0.00%        | 15.66%       | 90.81%      | 2.33%       | 68.87%      | 10.00% | 0.00%   | 11.11% | 63.49% | 1.39%  | 30.22% | 83.01%              | 92.27%              | 7.40%               | NA     | NA     | NA     | NA    | NA     | NA    | 68.93% | 0.00%   | 39.77% |
| preactresnet18     | ssba               | 92.88%     | 97.86%     | 1.99%      | 90.00% | 96.23%  | 3.53%  | 92.75%       | 93.83%       | 5.80%        | 92.44%      | 74.62%      | 23.39%      | 23.99% | 0.00%   | 26.31% | 63.49% | 5.38%  | 60.22% | 89.63%              | 90.50%              | 8.71%               | NA     | NA     | NA     | NA    | NA     | NA    | 74.24% | 0.53%   | 66.12% |
| preactresnet18     | wanet              | 91.25%     | 89.73%     | 9.73%      | 91.93% | 96.80%  | 3.06%  | 90.79%       | 76.99%       | 21.77%       | 93.47%      | 17.04%      | 78.33%      | 23.02% | 72.56%  | 8.67%  | 63.49% | 5.42%  | 61.02% | 91.94%              | 90.17%              | 9.37%               | NA     | NA     | NA     | NA    | NA     | NA    | 71.90% | 0.59%   | 75.18% |
| preactresnet18     | inputaware         | 90.67%     | 98.26%     | 1.66%      | 91.48% | 88.62%  | 10.61% | 90.59%       | 89.74%       | 9.82%        | 93.09%      | 1.72%       | 90.57%      | 17.72% | 53.40%  | 10.12% | 63.49% | 7.10%  | 53.20% | 91.39%              | 90.43%              | 9.13%               | NA     | NA     | NA     | NA    | NA     | NA    | 73.69% | 17.66%  | 56.22% |
| vgg19              | badnet             | 89.36%     | 95.93%     | 3.81%      | 86.25% | 94.37%  | 5.17%  | 18.40%       | 4.58%        | 11.28%       | 87.90%      | 21.28%      | 73.58%      | 10.00% | 100.00% | 0.00%  | 53.54% | 5.72%  | 52.86% | 88.07%              | 91.67%              | 7.78%               | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| vgg19              | blended            | 90.17%     | 99.12%     | 0.82%      | 87.58% | 98.13%  | 1.76%  | 19.19%       | 0.00%        | 11.28%       | 89.10%      | 93.31%      | 6.24%       | 10.00% | 100.00% | 0.00%  | 53.03% | 6.77%  | 48.69% | 89.18%              | 99.76%              | 0.23%               | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| vgg19              | sig                | 81.69%     | 99.80%     | 0.12%      | 79.60% | 99.94%  | 0.06%  | 50.56%       | 13.21%       | 8.74%        | 86.62%      | 98.47%      | 1.41%       | 10.00% | 100.00% | 0.00%  | 53.47% | 1.66%  | 33.93% | 80.27%              | 97.77%              | 1.89%               | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| vgg19              | ssba               | NA         | NA         | NA         | NA     | NA      | NA     | NA           | NA           | NA           | NA          | NA          | NA          | NA     | NA      | NA     | NA     | NA     | NA     | NA                  | NA                  | NA                  | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| vgg19              | wanet              | 88.43%     | 88.90%     | 10.29%     | 87.96% | 92.56%  | 7.02%  | 76.62%       | 0.00%        | 35.18%       | 91.04%      | 35.77%      | 59.81%      | 10.00% | 100.00% | 0.00%  | 52.03% | 7.53%  | 48.84% | 87.76%              | 96.24%              | 3.53%               | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| vgg19              | inputaware         | 10.00%     | 100.00%    | 0.00%      | NA     | NA      | NA     | 10.00%       | 100.00%      | 0.00%        | 10.00%      | 100.00%     | 0.00%       | 10.00% | 100.00% | 0.00%  | 53.05% | 0.00%  | 11.11% | NA                  | NA                  | NA                  | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| efficientnet_b3    | badnet             | 57.48%     | 9.51%      | 55.59%     | 68.13% | 78.46%  | 15.96% | NA           | NA           | NA           | 53.69%      | 5.09%       | 52.96%      | 10.29% | 84.74%  | 2.60%  | 13.85% | 16.16% | 13.46% | 73.83%              | 90.46%              | 7.61%               | 56.23% | 71.79% | 23.14% | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| efficientnet_b3    | blended            | 65.11%     | 87.76%     | 8.01%      | 71.09% | 96.82%  | 2.02%  | NA           | NA           | NA           | 61.16%      | 8.60%       | 44.66%      | 13.00% | 4.44%   | 13.87% | 13.85% | 8.99%  | 12.60% | 74.87%              | 94.74%              | 4.56%               | 50.92% | 99.79% | 0.12%  | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| efficientnet_b3    | sig                | 58.14%     | 99.71%     | 0.13%      | 63.86% | 99.40%  | 0.10%  | NA           | NA           | NA           | 56.92%      | 3.98%       | 24.03%      | 11.57% | 0.00%   | 12.76% | 13.85% | 26.26% | 12.48% | 70.39%              | 96.54%              | 2.46%               | NA     | NA     | NA     | NA    | NA     | NA    | 10.00% | 0.00%   | 11.11% |
| efficientnet_b3    | ssba               | 55.76%     | 24.40%     | 45.50%     | 10.00% | 100.00% | 0.00%  | NA           | NA           | NA           | 53.42%      | 7.10%       | 49.32%      | 14.75% | 2.01%   | 16.98% | 13.85% | 17.12% | 13.87% | 70.52%              | 74.03%              | 19.70%              | NA     | NA     | NA     | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| efficientnet_b3    | wanet              | 71.73%     | 7.93%      | 69.33%     | 77.26% | 43.63%  | 44.71% | NA           | NA           | NA           | 70.80%      | 3.59%       | 67.96%      | 16.52% | 12.46%  | 16.47% | 13.85% | 14.56% | 13.37% | 79.83%              | 86.93%              | 11.21%              | NA     | NA     | NA     | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| efficientnet_b3    | inputaware         | NA         | NA         | NA         | NA     | NA      | NA     | NA           | NA           | NA           | NA          | NA          | NA          | NA     | NA      | NA     | NA     | NA     | NA     | NA                  | NA                  | NA                  | NA     | NA     | NA     | NA    | NA     | NA    | NA     | NA      | NA     |
| mobilenet_v3_large | badnet             | 82.45%     | 93.39%     | 5.84%      | 77.95% | 72.04%  | 23.34% | NA           | NA           | NA           | 77.99%      | 3.00%       | 77.00%      | 21.25% | 17.42%  | 18.52% | 46.14% | 6.63%  | 45.67% | 79.95%              | 93.76%              | 5.54%               | 57.17% | 26.58% | 50.84% | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| mobilenet_v3_large | blended            | 83.43%     | 98.76%     | 1.13%      | 79.36% | 96.26%  | 2.94%  | NA           | NA           | NA           | 78.94%      | 12.27%      | 53.93%      | 13.80% | 24.59%  | 9.90%  | 46.14% | 7.41%  | 42.66% | 82.27%              | 97.23%              | 2.36%               | 52.14% | 99.39% | 0.50%  | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| mobilenet_v3_large | sig                | 77.13%     | 98.70%     | 1.04%      | 73.89% | 98.79%  | 1.00%  | NA           | NA           | NA           | 78.09%      | 2.51%       | 42.93%      | 11.62% | 0.00%   | 16.18% | 46.14% | 5.16%  | 37.11% | 74.02%              | 98.94%              | 0.77%               | 41.90% | 99.84% | 0.12%  | NA    | NA     | NA    | 10.00% | 0.00%   | 11.11% |
| mobilenet_v3_large | ssba               | 81.84%     | 85.06%     | 12.16%     | 77.70% | 62.86%  | 29.40% | NA           | NA           | NA           | 77.80%      | 6.94%       | 68.71%      | 17.60% | 25.73%  | 15.16% | 46.14% | 6.31%  | 45.23% | 80.56%              | 71.40%              | 23.79%              | 43.63% | 98.17% | 1.30%  | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| mobilenet_v3_large | wanet              | 78.00%     | 70.88%     | 24.80%     | 73.09% | 81.56%  | 15.23% | NA           | NA           | NA           | 80.40%      | 4.14%       | 77.22%      | 17.35% | 24.49%  | 16.41% | 46.14% | 7.13%  | 44.67% | 82.67%              | 26.06%              | 60.77%              | 58.87% | 18.39% | 53.73% | NA    | NA     | NA    | 10.00% | 100.00% | 0.00%  |
| mobilenet_v3_large | inputaware         | 77.35%     | 89.17%     | 9.12%      | 78.04% | 94.04%  | 5.01%  | NA           | NA           | NA           | 78.20%      | 7.09%       | 62.23%      | 17.89% | 20.23%  | 13.57% | 46.14% | 6.82%  | 41.62% | 79.74%              | 80.50%              | 16.63%              | 56.92% | 55.29% | 34.19% | NA    | NA     | NA    | 10.00% | 99.97%  | 0.00%  |
| densenet161        | badnet             | 84.33%     | 89.68%     | 9.10%      | 80.65% | 78.26%  | 18.30% | NA           | NA           | NA           | 83.59%      | 32.70%      | 56.87%      | 15.19% | 0.00%   | 16.81% | 49.29% | 7.00%  | 48.04% | 82.24%              | 90.20%              | 8.48%               | 60.99% | 15.98% | 57.10% | NA    | NA     | NA    | 46.41% | 0.00%   | 50.90% |
| densenet161        | blended            | 86.37%     | 98.79%     | 1.04%      | 81.78% | 94.83%  | 4.42%  | NA           | NA           | NA           | 84.39%      | 81.83%      | 14.74%      | 11.71% | 0.00%   | 13.40% | 49.29% | 8.82%  | 41.61% | 83.80%              | 97.22%              | 2.30%               | NA     | NA     | NA     | NA    | NA     | NA    | 51.22% | 0.19%   | 47.96% |
| densenet161        | sig                | 78.63%     | 98.67%     | 1.23%      | 76.39% | 97.70%  | 2.16%  | NA           | NA           | NA           | 83.49%      | 19.30%      | 49.21%      | 10.00% | 0.00%   | 11.19% | 49.29% | 2.83%  | 33.49% | 76.58%              | 94.59%              | 5.19%               | NA     | NA     | NA     | NA    | NA     | NA    | 34.68% | 0.39%   | 23.72% |
| densenet161        | ssba               | 84.18%     | 84.13%     | 13.39%     | 81.25% | 72.64%  | 22.49% | NA           | NA           | NA           | 83.76%      | 32.84%      | 55.90%      | 14.00% | 0.00%   | 15.12% | 49.29% | 8.63%  | 47.04% | 82.28%              | 86.88%              | 10.72%              | NA     | NA     | NA     | NA    | NA     | NA    | 51.61% | 0.68%   | 53.42% |
| densenet161        | wanet              | 84.61%     | 73.81%     | 22.72%     | 82.73% | 75.40%  | 20.92% | NA           | NA           | NA           | 87.52%      | 2.34%       | 84.48%      | 13.40% | 0.00%   | 14.76% | 49.29% | 6.39%  | 46.02% | 84.90%              | 71.19%              | 24.32%              | NA     | NA     | NA     | NA    | NA     | NA    | 37.08% | 0.34%   | 39.42% |
| densenet161        | inputaware         | 84.46%     | 94.41%     | 5.18%      | 85.02% | 93.70%  | 5.74%  | NA           | NA           | NA           | 86.77%      | 2.69%       | 82.02%      | 10.35% | 0.00%   | 11.67% | 49.29% | 11.37% | 39.77% | 85.12%              | 92.23%              | 6.62%               | NA     | NA     | NA     | NA    | NA     | NA    | 62.31% | 4.07%   | 51.01% |


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
