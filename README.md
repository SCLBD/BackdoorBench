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
  - 9 Backdoor defense methods: FT, [Spectral Signatures](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf), [AC](http://ceur-ws.org/Vol-2301/paper_18.pdf), [Fine-pruning](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13), [ABL](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf), [NAD](https://openreview.net/pdf?id=9l0K4OM-oXE), [NC](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), [DBD](https://arxiv.org/pdf/2202.03423.pdf), [ANP](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)
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

We present partial results on cifar10 with poison ratio = 1% here as an example. For complete results, please refer to our [leaderboard](https://backdoorbench.github.io/leaderboard-cifar10.html).

|                    | Backdoor Defense → | No Defense | No Defense | No Defense | AC     | AC     | AC     | Fine Tuning | Fine Tuning | Fine Tuning | ABL    | ABL     | ABL    | NAD    | NAD    | NAD    | Spectral Signatural | Spectral Signatural | Spectral Signatural |
| ------------------ | ------------------ | ---------- | ---------- | ---------- | ------ | ------ | ------ | ----------- | ----------- | ----------- | ------ | ------- | ------ | ------ | ------ | ------ | ------------------- | ------------------- | ------------------- |
| Targeted Model     | Backdoor Attack↓   | CA(%)      | ASR(%)     | RA(%)      | CA(%)  | ASR(%) | RA(%)  | CA(%)       | ASR(%)      | RA(%)       | CA(%)  | ASR(%)  | RA(%)  | CA(%)  | ASR(%) | RA(%)  | CA(%)               | ASR(%)              | RA(%)               |
| preactresnet18     | badnet             | 93.14%     | 74.73%     | 24.24%     | 90.99% | 56.01% | 42.01% | 93.07%      | 10.36%      | 84.58%      | 10.19% | 98.87%  | 0.20%  | 63.91% | 5.19%  | 63.31% | 91.36%              | 47.14%              | 50.00%              |
| preactresnet18     | blended            | 93.76%     | 94.88%     | 4.79%      | 90.23% | 94.36% | 5.12%  | 93.53%      | 91.96%      | 7.56%       | 17.34% | 0.00%   | 15.27% | 63.91% | 3.49%  | 53.93% | 91.28%              | 93.62%              | 5.58%               |
| preactresnet18     | sig                | 93.82%     | 83.40%     | 14.29%     | 90.66% | 90.26% | 8.66%  | 93.59%      | 88.17%      | 10.13%      | 10.00% | 98.42%  | 0.11%  | 63.91% | 0.97%  | 29.31% | 91.12%              | 74.29%              | 20.87%              |
| preactresnet18     | ssba               | 93.43%     | 73.44%     | 24.89%     | 91.33% | 55.07% | 40.53% | 93.24%      | 61.32%      | 35.98%      | 11.23% | 88.01%  | 2.82%  | 63.91% | 5.56%  | 60.18% | 92.36%              | 73.07%              | 24.73%              |
| preactresnet18     | wanet              | 90.65%     | 12.63%     | 79.94%     | 92.25% | 14.41% | 78.40% | 93.63%      | 0.67%       | 92.82%      | 17.72% | 52.21%  | 9.23%  | 63.91% | 5.42%  | 61.43% | 91.42%              | 39.59%              | 56.39%              |
| preactresnet18     | inputaware         | 91.74%     | 79.18%     | 19.89%     | 92.40% | 60.73% | 37.69% | 93.14%      | 77.66%      | 21.60%      | 10.21% | 98.01%  | 0.42%  | 63.91% | 8.88%  | 58.11% | 91.49%              | 70.79%              | 27.43%              |
| vgg19              | badnet             | 90.45%     | 83.98%     | 15.14%     | 88.39% | 66.18% | 31.47% | 89.12%      | 53.14%      | 44.14%      | 10.00% | 100.00% | 0.00%  | 53.30% | 6.23%  | 52.39% | 87.36%              | 63.31%              | 34.24%              |
| vgg19              | blended            | 90.47%     | 89.53%     | 9.62%      | 88.07% | 80.58% | 17.29% | 89.00%      | 76.28%      | 21.09%      | 10.00% | 100.00% | 0.00%  | 50.91% | 7.11%  | 45.84% | 89.50%              | 89.81%              | 9.49%               |
| vgg19              | sig                | 90.85%     | 83.77%     | 12.48%     | 88.32% | 93.03% | 5.33%  | 89.44%      | 69.87%      | 21.99%      | 10.00% | 100.00% | 0.00%  | 49.02% | 0.72%  | 29.06% | 89.04%              | 90.24%              | 7.69%               |
| vgg19              | ssba               | 89.96%     | 10.97%     | 79.72%     | 87.82% | 8.20%  | 80.93% | 89.08%      | 2.20%       | 85.46%      | 10.00% | 100.00% | 0.00%  | 53.39% | 5.42%  | 51.87% | 88.21%              | 5.44%               | 83.48%              |
| vgg19              | wanet              | 89.99%     | 1.71%      | 88.73%     | 89.82% | 2.39%  | 88.42% | 91.29%      | 1.10%       | 90.23%      | 10.00% | 100.00% | 0.00%  | 50.49% | 7.86%  | 46.96% | 90.30%              | 5.01%               | 86.60%              |
| vgg19              | inputaware         | 89.47%     | 58.09%     | 37.13%     | 89.84% | 61.44% | 34.74% | 90.57%      | 29.46%      | 61.39%      | 10.00% | 100.00% | 0.00%  | 51.71% | 6.02%  | 47.79% | 89.51%              | 64.68%              | 29.43%              |
| efficientnet_b3    | badnet             | 53.13%     | 3.80%      | 52.84%     | 69.75% | 3.36%  | 69.00% | 51.88%      | 4.94%       | 51.40%      | 13.92% | 39.96%  | 8.64%  | 24.35% | 3.28%  | 24.54% | 75.68%              | 2.12%               | 75.57%              |
| efficientnet_b3    | blended            | 61.57%     | 10.86%     | 53.18%     | 70.54% | 48.01% | 31.89% | 56.26%      | 6.31%       | 50.41%      | 9.50%  | 0.86%   | 9.92%  | 24.35% | 1.34%  | 24.90% | 76.14%              | 81.63%              | 12.90%              |
| efficientnet_b3    | sig                | 63.63%     | 3.87%      | 57.93%     | 71.85% | 74.46% | 13.78% | 58.24%      | 4.26%       | 53.17%      | 16.77% | 16.70%  | 14.93% | 24.35% | 2.43%  | 24.67% | 76.24%              | 78.27%              | 12.62%              |
| efficientnet_b3    | ssba               | 59.42%     | 4.49%      | 55.76%     | 68.81% | 7.59%  | 62.30% | 54.29%      | 5.03%       | 51.67%      | 13.61% | 35.71%  | 11.16% | 24.35% | 2.84%  | 25.01% | 72.18%              | 4.60%               | 68.24%              |
| efficientnet_b3    | wanet              | 73.07%     | 2.36%      | 72.02%     | 78.18% | 2.59%  | 76.51% | 70.67%      | 3.88%       | 67.89%      | 13.53% | 51.41%  | 7.56%  | 24.35% | 3.10%  | 24.41% | 82.56%              | 3.86%               | 79.19%              |
| efficientnet_b3    | inputaware         | 55.80%     | 5.49%      | 51.03%     | 75.01% | 55.42% | 30.40% | 57.43%      | 5.68%       | 52.10%      | 15.14% | 22.50%  | 12.32% | 24.35% | 2.79%  | 24.47% | 79.97%              | 65.71%              | 26.00%              |
| mobilenet_v3_large | badnet             | 84.46%     | 3.29%      | 83.43%     | 79.64% | 22.00% | 63.86% | 79.73%      | 2.86%       | 78.82%      | 17.82% | 24.02%  | 13.11% | 46.33% | 7.13%  | 45.54% | 74.94%              | 55.39%              | 34.03%              |
| mobilenet_v3_large | blended            | 83.97%     | 87.06%     | 10.17%     | 80.40% | 75.21% | 19.03% | 79.45%      | 34.66%      | 42.91%      | 20.24% | 5.06%   | 14.59% | 46.33% | 8.59%  | 41.78% | 80.53%              | 85.52%              | 11.61%              |
| mobilenet_v3_large | sig                | 84.48%     | 80.79%     | 10.77%     | 77.78% | 76.08% | 12.30% | 79.75%      | 51.12%      | 20.97%      | 17.80% | 0.00%   | 15.22% | 46.33% | 2.77%  | 34.29% | 79.49%              | 68.59%              | 15.22%              |
| mobilenet_v3_large | ssba               | 83.28%     | 9.71%      | 73.78%     | 78.32% | 16.37% | 63.36% | 79.84%      | 3.29%       | 73.70%      | 19.39% | 7.54%   | 17.86% | 46.33% | 5.97%  | 45.21% | 81.56%              | 21.91%              | 62.48%              |
| mobilenet_v3_large | wanet              | 80.86%     | 1.66%      | 80.02%     | 80.60% | 2.62%  | 79.04% | 81.41%      | 2.41%       | 79.14%      | 19.65% | 25.93%  | 13.77% | 46.33% | 8.13%  | 44.37% | 80.38%              | 6.60%               | 74.87%              |
| mobilenet_v3_large | inputaware         | 80.80%     | 90.72%     | 7.16%      | 81.42% | 87.99% | 10.88% | 80.19%      | 31.41%      | 46.18%      | 18.66% | 13.09%  | 14.84% | 46.33% | 5.00%  | 42.71% | 82.80%              | 63.79%              | 30.36%              |
| densenet161        | badnet             | 86.30%     | 16.32%     | 73.38%     | 81.93% | 16.01% | 70.99% | 85.26%      | 3.47%       | 82.79%      | 15.43% | 3.04%   | 16.03% | 48.70% | 7.20%  | 47.89% | 84.93%              | 2.99%               | 83.22%              |
| densenet161        | blended            | 86.65%     | 79.32%     | 16.94%     | 83.10% | 84.51% | 12.21% | 85.51%      | 61.74%      | 31.10%      | 15.99% | 0.03%   | 17.68% | 48.70% | 10.00% | 41.00% | 84.60%              | 4.42%               | 69.86%              |
| densenet161        | sig                | 86.63%     | 82.66%     | 13.49%     | 82.73% | 60.66% | 28.04% | 85.15%      | 75.18%      | 17.83%      | 16.50% | 0.00%   | 16.66% | 48.70% | 2.69%  | 32.54% | 84.76%              | 90.18%              | 7.87%               |
| densenet161        | ssba               | 86.31%     | 6.56%      | 79.24%     | 83.26% | 6.91%  | 75.90% | 85.06%      | 3.33%       | 80.96%      | 14.14% | 0.00%   | 15.94% | 48.70% | 8.73%  | 46.93% | 85.05%              | 2.89%               | 80.73%              |
| densenet161        | wanet              | 85.73%     | 2.17%      | 84.99%     | 85.34% | 2.78%  | 82.48% | 88.07%      | 1.17%       | 86.57%      | 15.03% | 0.00%   | 15.67% | 48.70% | 6.46%  | 46.59% | 85.77%              | 3.72%               | 81.63%              |
| densenet161        | inputaware         | 84.21%     | 66.37%     | 25.18%     | 84.68% | 75.03% | 20.23% | 87.28%      | 86.10%      | 11.42%      | 17.19% | 0.00%   | 18.02% | 48.70% | 12.03% | 39.62% | 85.74%              | 73.92%              | 21.10%              |

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
