# BackdoorBench: a comprehensive benchmark of backdoor attack and defense methods

![Pytorch 1.11.0](https://img.shields.io/badge/PyTorch-1.11-brightgreen) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .20](https://img.shields.io/badge/Release-2.0-brightgreen)

<p align="center">
<br>
  <a href="http://backdoorbench.com" target="_blank"> Website </a >  •  <a href="https://openreview.net/pdf?id=31_U7n18gM7"> Paper </a > •  <a href="http://backdoorbench.com/leader_cifar10"> Leaderboard </a > <br>
<br>
</p >

BackdoorBench is a comprehensive benchmark of backdoor learning, which studies the adversarial vulnerablity of deep learning models in the training stage. It aims to provide **easy implementations** of mainstream backdoor attack and defense methods.

### ❗V2.0 Updates
> ✅ **Correction**:
>   1. **Attack** : Fix the bug in [Label Consistent](./attack/lc.py) attack method, in v1.0 version, poisoned data only add adversarial noise without square trigger, which is not consistent with the paper.
> 
> ✅ **Code**: 
>    1. **Structure** : Warp attack methods and defense methods into classes and reduce replicated code.
>    2. **Dataset Processing** : Update bd_dataset into bd_dataset_v2, which can handle large scale dataset more efficently.
>    3. **Poison Data Generation** : Provide necessary code to generate poisoned dataset for attack methods (see ./resource folder, we have seperate readme files).
>    4. **Models** : We add VGG19_bn, ConvNeXT_tiny, ViT_B_16.
>    
> ✅ **Methods**: 
>    1. **Attack** :Add 4 new attack methods: [Blind](./attack/blind.py), [BPP](./attack/bpp.py), [LIRA](./attack/lira.py), [TrojanNN](./attack/trojannn.py). (Totally 12 attack methods now).
>    2. **Defense** :Add 6 new defense methods: [CLP](./defense/clp.py), [D-BR](./defense/d-br.py), [D-ST](./defense/d-st.py), [EP](./defense/ep.py), [I-BAU](./defense/i-bau.py), [BNP](./defense/bnp.py). (Totally 15 defense methods now).
>    
> ✅ **Analysis Tools** : 
>    1. **Data Analysis** : Add 2 new methods: [UMAP](./analysis/visual_umap.py), [Image Quality](./analysis/visual_quality.py)
>    2. **Models Analysis** : Add 9 new methods: [Activated Image](./analysis/visual_act.py), [Feature Visualization](./analysis/visual_fv.py), [Feature Map](./analysis/visual_fm.py), [Activation Distribution](./analysis/visual_actdist.py), [Trigger Activation Change](./analysis/visual_tac.py), [Lipschitz Constant](./analysis/visual_lips.py), [Loss Landscape](./analysis/visual_landscape.py), [Network Structure](./analysis/visual_network.py), [Eigenvalues of Hessian](./analysis/visual_hessian.py)
>    3. **Evaluation Analysis** : Add 2 new methods: [Confusion Matrix](./analysis/visual_cm.py), [Metric](./analysis/visual_metric.py)
>  
> 🔲 Comprehensive evaluations will be coming soon...

### ❗ For V1.0 please check [here](https://github.com/SCLBD/BackdoorBench/tree/v1)

<font size=5><center><b> Table of Contents </b> </center></font>

* [Features](#features)

* [Installation](#Installation)

* [Quick Start](#quick-start)

  * [Attack](#attack)
  
  * [Defense](#defense)

* [Supported attacks](#supported-attacks)

* [Supported defenses](#supported-defsense)

* [Analysis Tools](#analysis-tools)

* [Citation](#citation)

* [Copyright](#copyright)

---


## Features
<a href="#top">[Back to top]</a>

BackdoorBench has the following features:

⭐️  **Methods**:
  - 12 Backdoor attack methods: [BadNets](./attack/badnet.py), [Blended](./attack/blended.py), [Blind](./attack/blind.py), [BPP](./attack/bpp.py), [Input-aware](./attack/inputaware.py), [Label Consistent](./attack/lc.py), [Low Frequency](./attack/lf.py), [LIRA](./attack/lira.py), [SIG](./attack/sig.py), [SSBA](./attack/ssba.py), [TrojanNN](./attack/trojannn.py), [WaNet](./attack/wanet.py)
  - 15 Backdoor defense methods:  [FT](./defense/ft.py), [Spectral](./defense/spectral.py), [AC](./defense/ac.py), [FP](./defense/fp.py), [ABL](./defense/abl.py), [NAD](./defense/nad.py), [NC](nc), [DBD]((./defense/dbd.py)), [ANP](./defense/anp.py),[CLP](./defense/clp.py), [D-BR](./defense/d-br.py), [D-ST](./defense/d-st.py), [EP](./defense/ep.py), [I-BAU](./defense/i-bau.py), [BNP](./defense/bnp.py)

⭐️ **Datasets**: CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet 

⭐️ **Models**: PreAct-Resnet18, VGG19_bn, ConvNeXT_tiny, ViT_B_16, VGG19, DenseNet-161, MobileNetV3-Large, EfficientNet-B3


⭐️ **Learboard**: We provide a [**public leaderboard**](http://backdoorbench.com/leader_cifar10) of evaluating all backdoor attacks against all defense methods.

BackdoorBench will be continuously updated to track the lastest advances of backddor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. **You are welcome to contribute your backdoor methods to BackdoorBench.**



## Installation

<a href="#top">[Back to top]</a>

You can run the following script to configurate necessary environment

```
git clone git@github.com:SCLBD/BackdoorBench.git
cd BackdoorBench
conda create -n backdoorbench python=3.8
conda activate backdoorbench
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

## Quick Start

### Attack

<a href="#top">[Back to top]</a>

This is a example for BadNets

1. Generate trigger

If you want to change the trigger for BadNets, you should go to the `./resource/badnet`, and follow the readme there to generate new trigger pattern.
```shell
python ./resource/badnet/generate_white_square.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --output_path ./resource/badnet/trigger_image.png
```
Note that for data-poisoning-based attacks (BadNets, Blended, Label Consistent, Low Frequency, SSBA). 
Our scripts in `./attack` are just for training, they do not include the data generation process.(Because they are time-comsuming, and we do not want to waste your time.) 
You should go to the `./resource` folder to generate the trigger for training.

2. Backdoor training
```
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml --patch_mask_path ../resource/badnet/trigger_image.png  --save_folder_name badnet_0_1
```
After attack you will get a folder with all files saved in `./record/<folder name in record>`, including `attack_result.pt` for attack model and backdoored data, which will be used by following defense methods.
If you want to change the args, you can both specify them in command line and in corresponding YAML config file (eg. [default.yaml](./config/attack/badnet/default.yaml)).(They are the defaults we used if no args are specified in command line.)
The detailed descriptions for each attack may be put into the `add_args` function in each script.

Note that for some attacks, they may need pretrained models to generate backdoored data. For your ease, we provide various data/trigger/models we generated in google drive. You can download them at [here](https://drive.google.com/drive/folders/1lnCObVBIUTSlLWIBQtfs_zi7W8yuvR-2?usp=share_link)




### Defense

<a href="#top">[Back to top]</a>

This is a demo script of running abl defense on cifar-10 for badnet attack. Before defense you need to run badnet attack on cifar-10 at first. Then you use the folder name as result_file.

```
python ./defense/abl.py --result_file badnet_0_1 --yaml_path ./config/defense/abl/cifar10.yaml --dataset cifar10
```


If you want to change the args, you can both specify them in command line and in corresponding YAML config file (eg. [default.yaml](./config/defense/abl/default.yaml)).(They are the defaults we used if no args are specified in command line.)
The detailed descriptions for each attack may be put into the `add_args` function in each script.

## Supported attacks

<a href="#top">[Back to top]</a>

|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BadNets          | [badnet.py](./attack/badnet.py)         | [BadNets: Identifying Vulnerabilities in  the Machine Learning Model Supply Chain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j) IEEE Access 2019 |
| Blended          | [blended.py](./attack/blended.py)       | [Targeted Backdoor Attacks on Deep  Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526v1) Arxiv 2017                                                                                                                                                                                                                                     |
| Blind            | [blind.py](./attack/blind.py)           | [Blind Backdoors in Deep Learning Models](https://www.cs.cornell.edu/~shmat/shmat_usenix21blind.pdf) USENIX 2021                                                                                                                                                                                                                                              |
| BPP              | [bpp.py](./attack/bpp.py)               | [BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.pdf) CVPR 2022                                            |
| Input-aware      | [inputaware.py](./attack/inputaware.py) | [Input-Aware Dynamic Backdoor Attack](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf) NeurIPS 2020                                                                                                                                                                                                                 |
| Label Consistent | [lc.py](./attack/lc.py)                 | [Label-Consistent Backdoor Attacks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjvwKTx2bH4AhXCD0QIHVMWApkQFnoECAsQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1912.02771&usg=AOvVaw0NbPR9lguGTsEn3ZWtPBDR) Arxiv 2019                                                                                                 |
| Low Frequency    | [lf.py](./attack/lf.py)                 | [Rethinking the Backdoor Attacks’ Triggers: A Frequency Perspective](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf) ICCV2021                                                                                                                                |
| LIRA             | [lira.py](./attack/lira.py)             | [LIRA: Learnable, Imperceptible and Robust Backdoor Attacks](https://openaccess.thecvf.com/content/ICCV2021/papers/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                               |
| SIG              | [sig.py](./attack/sig.py)               | [A new backdoor attack in  cnns by training set corruption](https://ieeexplore.ieee.org/document/8802997) ICIP 2019                                                                                                                                                                                                                                           |
| SSBA             | [ssba.py](./attack/ssba.py)             | [Invisible Backdoor Attack with  Sample-Specific Triggers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                                    |
| TrojanNN         | [trojannn.py](./attack/trojannn.py)     | [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech) NDSS 2018                                                                                                                                                                                               |
| WaNet            | [wanet.py](./attack/wanet.py)           | [WaNet -- Imperceptible  Warping-Based Backdoor Attack](https://openreview.net/pdf?id=eEn8KTtJOx) ICLR 2021                                                                                                                                                                                                                                                   |

## Supported defenses 

<a href="#top">[Back to top]</a>

|       | File name                 | Paper                |
| :------------- |:-------------|:-----|
| FT| [ft.py](./defense/ft.py) | standard fine-tuning|
| FP | [fp.py](./defense/fp.py) | [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) RAID 2018 |
| NAD      | [nad.py](./defense/nad.py)                | [Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks](https://openreview.net/pdf?id=9l0K4OM-oXE) ICLR 2021 |
| NC       | [nc.py](./defense/nc.py)                   | [Neural Cleanse: Identifying And Mitigating Backdoor Attacks In Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), IEEE S&P 2019 |
| ANP      | [anp.py](./defense/anp.py)                | [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf) NeurIPS 2021 |
| AC       | [ac.py](./defense/ac.py)                   | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](http://ceur-ws.org/Vol-2301/paper_18.pdf) ceur-ws 2018 |
| Spectral | [spectral.py](./defense/spectral.py) | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf) NeurIPS 2018 |
| ABL      | [abl.py](./defense/abl.py)                | [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf) NeurIPS 2021 |
| DBD | [dbd.py](./defense/dbd.py) | [Backdoor Defense Via Decoupling The Training Process](https://arxiv.org/pdf/2202.03423.pdf) ICLR 2022 |
| CLP | [clp.py](./defense/clp.py) | [Data-free backdoor removal based on channel lipschitzness](https://arxiv.org/pdf/2208.03111.pdf) ECCV 2022 |
| I-BAU | [i-bau.py](./defense/i-bau.py) | [Adversarial unlearning of backdoors via implicit hypergradient](https://arxiv.org/pdf/2110.03735.pdf) ICLR 2022 |
| D-BR,D-ST | [d-br.py](./defense/d-br.py) [d-st.py](./defense/d-st.py) | [Effective backdoor defense by exploiting sensitivity of poisoned samples](https://proceedings.neurips.cc/paper_files/paper/2022/file/3f9bbf77fbd858e5b6e39d39fe84ed2e-Paper-Conference.pdf) NeurIPS 2022 |
| EP,BNP | [ep.py](./defense/ep.py) [bnp.py](./defense/bnp.py) | [Pre-activation Distributions Expose Backdoor Neurons](https://proceedings.neurips.cc/paper_files/paper/2022/file/76917808731dae9e6d62c2a7a6afb542-Paper-Conference.pdf) NeurIPS 2022 |








<a href="#top">[Back to top]</a>
### Analysis Tools


|        File name                                    |          Method                                                                 |             Category           |
|:----------------------------------------------------|:--------------------------------------------------------------------------------|:--------------------------------|
| [visual_tsne.py](analysis/visual_tsne.py)           | T-SNE, the T-SNE of features                                                    |            Data Analysis       |
| [visual_umap.py](analysis/visual_umap.py)           | UMAP, the UMAP of features                                                      |            Data Analysis       |
| [visual_quality.py](./analysis/visual_quality.py)   | Image Quality, evaluating the given results using some image quality metrics    |            Data Analysis       |
| [visual_na.py](analysis/visual_na.py)               | Neuron Activation, the activation value of a given layer of Neurons             |            Model Analysis      |
| [visual_shap.py](analysis/visual_shap.py)           | Shapely Value, the Shapely Value for given inputs and a given layer             |            Model Analysis      |
| [visual_gradcam.py](analysis/visual_gradcam.py)     | Grad-CAM, the Grad-CAM for given inputs and a given layer                       |            Model Analysis      |
| [visualize_fre.py](analysis/visualize_fre.py)       | Frequency Map, the Frequency Saliency Map for given inputs and a given layer    |            Model Analysis      |
| [visual_act.py](analysis/visual_act.py)             | Activated Image, the top images who activate the given layer of Neurons most    |            Model Analysis      |
| [visual_fv.py](analysis/visual_fv.py)               | Feature Visualization, the synthetic images which activate the given Neurons    |            Model Analysis      |
| [visual_fm.py](analysis/visual_fm.py)               | Feature Map, the output of a given layer of CNNs for a given image              |            Model Analysis      |
| [visual_actdist.py](analysis/visual_actdist.py)     | Activation Distribution, the class distribution of Top-k images which activate the Neuron most    |            Model Analysis      |
| [visual_tac.py](analysis/visual_tac.py)             | Trigger Activation Change, the average (absolute) activation change between images with and without triggers   |            Model Analysis      |
| [visual_lips.py](analysis/visual_lips.py)           | Lipschitz Constant, the lipschitz constant of each neuron                       |            Model Analysis      |
| [visual_landscape.py](analysis/visual_landscape.py) | Loss Landscape, the loss landscape of given results with two random directions  |            Model Analysis      |
| [visual_network.py](analysis/visual_network.py)     | Network Structure, the Network Structure of given model                         |            Model Analysis      |
| [visual_hessian.py](analysis/visual_hessian.py)     | Eigenvalues of Hessian, the dense plot of hessian matrix for a batch of data    |            Model Analysis      |
| [visual_metric.py](analysis/visual_metric.py)       | Metrics, evaluating the given results using some metrics                        |              Evaluation        |
| [visual_cm.py](analysis/visual_cm.py)               | Confusion Matrix          | |








































## Citation

<a href="#top">[Back to top]</a>

If interested, you can read our recent works about backdoor learning, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}

@article{wu2023adversarial,
  title={Adversarial Machine Learning: A Systematic Survey of Backdoor Attack, Weight Attack and Adversarial Example},
  author={Wu, Baoyuan and Liu, Li and Zhu, Zihao and Liu, Qingshan and He, Zhaofeng and Lyu, Siwei},
  journal={arXiv preprint arXiv:2302.09457},
  year={2023}
}

@article{cheng2023tat,
  title={TAT: Targeted backdoor attacks against visual object tracking},
  author={Cheng, Ziyi and Wu, Baoyuan and Zhang, Zhenya and Zhao, Jianjun},
  journal={Pattern Recognition},
  volume={142},
  pages={109629},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{sensitivity-backdoor-defense-nips2022,
 title = {Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples},
 author = {Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
 booktitle = {Advances in Neural Information Processing Systems},
 volume = {35},
 pages = {9727--9737},
 year = {2022}
}

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


## Copyright

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) and [Shenzhen Research Institute of Big Data](http://www.sribd.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
