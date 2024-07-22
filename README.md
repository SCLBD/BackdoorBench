<img src="resource/pyg_logo.png" style="height: 60px;" align="right">

# BackdoorBench: a comprehensive benchmark of backdoor attack and defense methods

![Pytorch 1.11.0](https://img.shields.io/badge/PyTorch-1.11-brightgreen) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .20](https://img.shields.io/badge/Release-2.0-brightgreen)

<p align="center">
<br>
  <a href="http://backdoorbench.com" target="_blank"> Website </a >  •  <a href="https://openreview.net/pdf?id=31_U7n18gM7"> Paper </a > • <a href="http://backdoorbench.com/doc/index"> Doc </a > • <a href="http://backdoorbench.com/leader_cifar10"> Leaderboard </a > <br>
<br>
</p >

BackdoorBench is a comprehensive benchmark of backdoor learning, which studies the adversarial vulnerablity of deep learning models in the training stage. It aims to provide **easy implementations** of mainstream backdoor attack and defense methods.

### ❗Model and Data Updates

We disclose the backdoor model we used and the corresponding backdoor attack image in the link below. Each zip file contains the following things: 

- **bd_train_dataset**: train backdoor data
- **bd_test_dataset**: test backdoor data
- **attack_result.py**: the backdoor model and the module that reads data
- **cross_test_dataset**: cross mode data during training(for some special backdoor attack: wanet, inputaware and so on)

If you want to use the backdoor model, you can download the zip file and unzip in your own workspace. Then you can use the function load_attack_result in the file [save_load_attack.py](./utils/save_load_attack.py) to load the backdoor model, the poisoned train data and the poisoned test data.

[Backdoor Model](https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc)

✅ Since the previously shared document was inaccessible, we have now re-shared the backdoor model, and the link has been updated for your convenience. You can directly download the model using the link provided above.

### ❗V2.2 Updates
> ✅ **Methods**:  
>    1. **Defense** :Add 9 new defense/detection methods: [STRIP](./detection_pretrain/strip.py), [BEATRIX](./detection_pretrain/beatrix.py), [SCAN](./detection_pretrain/scan.py), [SPECTRE](./detection_pretrain/spectre.py), [SS](./detection_pretrain/spectral.py), [AGPD](./detection_pretrain/agpd.py), [SentiNet](./detection_infer/sentinet.py), [STRIP](./detection_infer/strip.py), [TeCo](./detection_infer/teco.py) (Totally 28 defense/detection methods now). 
>


### ❗ For V2.0 please check [here](https://github.com/SCLBD/BackdoorBench/tree/v2) 

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
  - 16 Backdoor attack methods: [BadNets](./attack/badnet.py), [Blended](./attack/blended.py), [Blind](./attack/blind.py), [BppAttack](./attack/bpp.py), [CTRL](./attack/ctrl.py), [FTrojan](./attack/ftrojann.py), [Input-aware](./attack/inputaware.py), [LC](./attack/lc.py), [LF](./attack/lf.py), [LIRA](./attack/lira.py), [PoisonInk](./attack/poison_ink.py), [ReFool](./attack/refool.py), [SIG](./attack/sig.py), [SSBA](./attack/ssba.py), [TrojanNN](./attack/trojannn.py), [WaNet](./attack/wanet.py)
  - 28 Backdoor defense/detection methods: [ABL](./defense/abl.py), [AC](./defense/ac.py), [ANP](./defense/anp.py), [CLP](./defense/clp.py), [D-BR](./defense/d-br.py), [D-ST](./defense/d-st.py), [DBD](./defense/dbd.py), [EP](./defense/ep.py), [BNP](./defense/bnp.py), [FP](./defense/fp.py), [FT](./defense/ft.py), [FT-SAM](./defense/ft-sam.py), [I-BAU](./defense/i-bau.py), [MCR](./defense/mcr.py), [NAB](./defense/nab.py), [NAD](./defense/nad.py), [NC](./defense/nc.py), [NPD](./defense/npd.py), [RNP](./defense/rnp.py), [SAU](./defense/sau.py), [SS](./defense/spectral.py), [STRIP](./detection_pretrain/strip.py), [BEATRIX](./detection_pretrain/beatrix.py), [SCAN](./detection_pretrain/scan.py), [SPECTRE](./detection_pretrain/spectre.py), [SS](./detection_pretrain/spectral.py), [AGPD](./detection_pretrain/agpd.py), [SentiNet](./detection_infer/sentinet.py), [STRIP](./detection_infer/strip.py), [TeCo](./detection_infer/teco.py)

⭐️ **Datasets**: CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet 

⭐️ **Models**: PreAct-Resnet18, VGG19_bn, ConvNeXT_tiny, ViT_B_16, VGG19, DenseNet-161, MobileNetV3-Large, EfficientNet-B3


⭐️ **Learboard**: We provide a [**public leaderboard**](http://backdoorbench.com/leader_cifar10) of evaluating all backdoor attacks against all defense methods.

BackdoorBench will be continuously updated to track the lastest advances of backddor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. **You are welcome to contribute your backdoor methods to BackdoorBench.**



## Installation

<a href="#top">[Back to top]</a>

You can run the following script to configure the necessary environment.

```
git clone git@github.com:SCLBD/BackdoorBench.git
cd BackdoorBench
conda create -n backdoorbench python=3.8
conda activate backdoorbench
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

You can also download backdoorbench by pip.

```
pip install -i https://test.pypi.org/simple/ backdoorbench
```

The pip version of backdoorbench can be viewed at this [link](https://github.com/SCLBD/bdzoo2-pip)

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

Note that for some attacks, they may need pretrained models to generate backdoored data. For your ease, we provide various data/trigger/models we generated in google drive. You can download them at [here](https://drive.google.com/drive/folders/1lnCObVBIUTSlLWIBQtfs_zi7W8yuvR-2?usp=share_link) (including **clean_model** files, **ssba,lf,lc generated triggers/samples** for you convenience.)




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

|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BadNets          | [badnet.py](./attack/badnet.py)         | [BadNets: Identifying Vulnerabilities in  the Machine Learning Model Supply Chain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j) IEEE Access 2019                                                                         |
| Blended          | [blended.py](./attack/blended.py)       | [Targeted Backdoor Attacks on Deep  Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526v1) Arxiv 2017                                                                                                                                                                                                                                                                                                             |
| Blind            | [blind.py](./attack/blind.py)           | [Blind Backdoors in Deep Learning Models](https://www.cs.cornell.edu/~shmat/shmat_usenix21blind.pdf) USENIX 2021                                                                                                                                                                                                                                                                                                                      |
| BPP              | [bpp.py](./attack/bpp.py)               | [BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.pdf) CVPR 2022                                                                                                                    |
| CTRL             | [ctrl.py](./attack/ctrl.py)             | [An Embarrassingly Simple Backdoor Attack on Self-supervised Learning](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiB6pfnu7KDAxWiaPUHHSzeDXIQFnoECAsQAQ&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2023%2Fpapers%2FLi_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.pdf&usg=AOvVaw2rR9-Se-bZgF3U0EU4puPE&opi=89978449)  ICCV 2023 |
| FTrojan          | [ftrojann.py](./attack/ftrojann.py)     | [An Invisible Black-box Backdoor Attack through Frequency Domain](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730396.pdf)   ECCV 2022                                                                                                                                                                                                                                                                                 |
| Input-aware      | [inputaware.py](./attack/inputaware.py) | [Input-Aware Dynamic Backdoor Attack](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf) NeurIPS 2020                                                                                                                                                                                                                                                                                         |
| Label Consistent | [lc.py](./attack/lc.py)                 | [Label-Consistent Backdoor Attacks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjvwKTx2bH4AhXCD0QIHVMWApkQFnoECAsQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1912.02771&usg=AOvVaw0NbPR9lguGTsEn3ZWtPBDR) Arxiv 2019                                                                                                 |
| Low Frequency    | [lf.py](./attack/lf.py)                 | [Rethinking the Backdoor Attacks’ Triggers: A Frequency Perspective](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf) ICCV2021                                                                                                                                |
| LIRA             | [lira.py](./attack/lira.py)             | [LIRA: Learnable, Imperceptible and Robust Backdoor Attacks](https://openaccess.thecvf.com/content/ICCV2021/papers/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                                                                                                       |
| PoisonInk        | [poison_ink.py](./attack/poison_ink.py) | [Poison ink: Robust and invisible backdoor attack](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9870671)  IEEE Transactions on Image Processing, 2022                                                                                                                                                                                                                                                                     |
| ReFool           | [refool.py](./attack/refool.py)         | [Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiioOK2vLKDAxU4iq8BHbPGDEoQFnoECAsQAQ&url=https%3A%2F%2Fwww.ecva.net%2Fpapers%2Feccv_2020%2Fpapers_ECCV%2Fpapers%2F123550188.pdf&usg=AOvVaw2_cqNKyWBEfXSBhaW5IOMj&opi=89978449)  ECCV 2020                                                                        |
| SIG              | [sig.py](./attack/sig.py)               | [A new backdoor attack in  cnns by training set corruption](https://ieeexplore.ieee.org/document/8802997) ICIP 2019                                                                                                                                                                                                                                                                                                                   |
| SSBA             | [ssba.py](./attack/ssba.py)             | [Invisible Backdoor Attack with  Sample-Specific Triggers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                                                                                                            |
| TrojanNN         | [trojannn.py](./attack/trojannn.py)     | [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech) NDSS 2018                                                                                                                                                                                                                                                                                                          |
| WaNet            | [wanet.py](./attack/wanet.py)           | [WaNet -- Imperceptible  Warping-Based Backdoor Attack](https://openreview.net/pdf?id=eEn8KTtJOx) ICLR 2021                                                                                                                                                                                                                                                                                                                           |

## Supported defenses 

<a href="#top">[Back to top]</a>

|           | File name                                                 | Paper                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:----------|:----------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ABL       | [abl.py](./defense/abl.py)                                | [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf) NeurIPS 2021                                                                                                                                                                                                                                                                                  |
| AC        | [ac.py](./defense/ac.py)                                  | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](http://ceur-ws.org/Vol-2301/paper_18.pdf) ceur-ws 2018                                                                                                                                                                                                                                                                                                                      |
| ANP       | [anp.py](./defense/anp.py)                                | [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf) NeurIPS 2021                                                                                                                                                                                                                                                                                      |
| CLP       | [clp.py](./defense/clp.py)                                | [Data-free backdoor removal based on channel lipschitzness](https://arxiv.org/pdf/2208.03111.pdf) ECCV 2022                                                                                                                                                                                                                                                                                                                                               |
| D-BR,D-ST | [d-br.py](./defense/d-br.py) [d-st.py](./defense/d-st.py) | [Effective backdoor defense by exploiting sensitivity of poisoned samples](https://proceedings.neurips.cc/paper_files/paper/2022/file/3f9bbf77fbd858e5b6e39d39fe84ed2e-Paper-Conference.pdf) NeurIPS 2022                                                                                                                                                                                                                                                 |
| DBD       | [dbd.py](./defense/dbd.py)                                | [Backdoor Defense Via Decoupling The Training Process](https://arxiv.org/pdf/2202.03423.pdf) ICLR 2022                                                                                                                                                                                                                                                                                                                                                    |
| EP,BNP    | [ep.py](./defense/ep.py) [bnp.py](./defense/bnp.py)       | [Pre-activation Distributions Expose Backdoor Neurons](https://proceedings.neurips.cc/paper_files/paper/2022/file/76917808731dae9e6d62c2a7a6afb542-Paper-Conference.pdf) NeurIPS 2022                                                                                                                                                                                                                                                                     |
| FP        | [fp.py](./defense/fp.py)                                  | [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) RAID 2018                                                                                                                                                                                                                                                                                                   |
| FT        | [ft.py](./defense/ft.py)                                  | standard fine-tuning                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| FT-SAM    | [ft-sam.py](./defense/ft-sam.py)                          | [Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjGlOzGuLKDAxW2j68BHQ1cDKoQFnoECAsQAQ&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2023%2Fpapers%2FZhu_Enhancing_Fine-Tuning_Based_Backdoor_Defense_with_Sharpness-Aware_Minimization_ICCV_2023_paper.pdf&usg=AOvVaw3j_4UcalC7moFEDuHaLXjO&opi=89978449) ICCV 2023 |
| I-BAU     | [i-bau.py](./defense/i-bau.py)                            | [Adversarial unlearning of backdoors via implicit hypergradient](https://arxiv.org/pdf/2110.03735.pdf) ICLR 2022                                                                                                                                                                                                                                                                                                                                          |
| MCR       | [mcr.py](./defense/mcr.py)                                | [Bridging mode connectivity in loss landscapes and adversarial robustness](https://openreview.net/pdf?id=SJgwzCEKwH) ICLR 2020                                                                                                                                                                                                                                                                                                                            |
| NAB       | [nab.py](./defense/nab.py)                                | [Beating Backdoor Attack at Its Own Game](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiT1P3LubKDAxWBdvUHHZU0C_4QFnoECAgQAQ&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2023%2Fpapers%2FLiu_Beating_Backdoor_Attack_at_Its_Own_Game_ICCV_2023_paper.pdf&usg=AOvVaw2q9z7lRkjVriRnqJCfacLZ&opi=89978449)   ICCV 2023                                                                             |
| NAD       | [nad.py](./defense/nad.py)                                | [Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks](https://openreview.net/pdf?id=9l0K4OM-oXE) ICLR 2021                                                                                                                                                                                                                                                                                                                 |
| NC        | [nc.py](./defense/nc.py)                                  | [Neural Cleanse: Identifying And Mitigating Backdoor Attacks In Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), IEEE S&P 2019                                                                                                                                                                                                                                                                                             |
| NPD       | [npd.py](./defense/npd.py)                                | [Neural polarizer: A lightweight and effective backdoor defense via purifying poisoned features](https://openreview.net/pdf?id=VFhN15Vlkj) NeurIPS 2023                                                                                                                                                                                                                                                                                                   |
| RNP       | [rnp.py](./defense/rnp.py)                                | [Reconstructive Neuron Pruning for Backdoor Defense](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi0htKPurKDAxW9aPUHHW3lC8kQFnoECAsQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F2021%2Ffile%2F8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf&usg=AOvVaw2Se3o40f41RV5M8xzgqjvh&opi=89978449) NeurIPS 2021                                                                                             |                                                                                                         
| SAU       | [sau.py](./defense/sau.py)                                 | [Shared adversarial unlearning: Backdoor mitigation by unlearning shared adversarial examples](https://openreview.net/pdf?id=zqOcW3R9rd) NeurIPS 2023                                                                                                                                                                                                                                                                                                                                                         |
| SS         | [spectral.py](./defense/spectral.py)                      | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf) NeurIPS 2018                                                                                                                                                                                                                                                                                                         |

## Supported Detection 

<a href="#top">[Back to top]</a>

### pretrain

|         | File name                                    | Paper                                                                                                                                                                                                                                                                                                                                               |
|:--------|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| STRIP   | [strip.py](./detection_pretrain/strip.py)       | [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3359789.3359790) ACSAC 2019                                                                                                                                                                                                                    |
| BEATRIX | [beatrix.py](./detection_pretrain/beatrix.py)   | [The Beatrix Resurrections: Robust Backdoor Detection via Gram Matrices](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwibhPKPwLKDAxXFia8BHUp2CmEQFnoECCIQAQ&url=https%3A%2F%2Fwww.usenix.org%2Fsystem%2Ffiles%2Fusenixsecurity23-pan.pdf&usg=AOvVaw0GLq_ZUB3OK1AoKhW9TNx_&opi=89978449)     NDSS 2023 |
| SCAN    | [scan.py](./detection_pretrain/scan.py)         | [Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection](https://www.usenix.org/system/files/sec21-tang-di.pdf)   USENIX Security 21                                                                                                                                                                        |
| SPECTRE | [spectre.py](./detection_pretrain/spectre.py)   | [SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics](https://par.nsf.gov/servlets/purl/10268374) ICML 2020                                                                                                                                                                                                                         |
| SS      | [spectral.py](./detection_pretrain/spectral.py) | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf)               NeurIPS 2018                                                                                                                                                                                        |
| AGPD    | [agpd.py](./detection_pretrain/agpd.py)             | [Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks](https://arxiv.org/abs/2312.06230) arXiv |

### inference-time

|       | File name                              | Paper                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|:------|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SentiNet | [sentinet.py](./detection_infer/sentinet.py) | [Sentinet: Detecting localized universal attacks against deep learning systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9283822)  2020 IEEE Security and Privacy Workshops (SPW) |
| STRIP | [strip.py](./detection_infer/strip.py) | [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3359789.3359790) ACSAC 2019                                                                                                                                                                                                                                                                                                                                     |
| TeCo  | [teco.py](./detection_infer/teco.py)   | [Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjyktXTv7KDAxUZft4KHeeTC5wQFnoECAwQAQ&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FCVPR2023%2Fpapers%2FLiu_Detecting_Backdoors_During_the_Inference_Stage_Based_on_Corruption_Robustness_CVPR_2023_paper.pdf&usg=AOvVaw38J5yiw9xqwcRNAyeZB1QF&opi=89978449)  CVPR 2023 |














[//]: # "| AGPD  | [agpd.py]&#40;./detection_infer/agpd.py&#41;   | |"

## Model and Data Downloading
<a href="#top">[Back to top]</a>

We disclose the backdoor model we used and the corresponding backdoor attack image in the link below. Each zip file contains the following things: 

- **bd_train_dataset**: train backdoor data
- **bd_test_dataset**: test backdoor data
- **attack_result.py**: the backdoor model and the module that reads data
- **cross_test_dataset**: cross mode data during training(for some special backdoor attack: wanet, inputaware and so on)

If you want to use the backdoor model, you can download the zip file and unzip in your own workspace. Then you can use the function load_attack_result in the file [save_load_attack.py](./utils/save_load_attack.py) to load the backdoor model, the poisoned train data and the poisoned test data.

We provide the whole sharepoint link at [Backdoor Model](https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc) for you to download the data and model.  






### Analysis Tools
<a href="#top">[Back to top]</a>

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

@article{wu2023defenses,
  title={Defenses in Adversarial Machine Learning: A Survey},
  author={Wu, Baoyuan and Wei, Shaokui and Zhu, Mingli and Zheng, Meixi and Zhu, Zihao and Zhang, Mingda and Chen, Hongrui and Yuan, Danni and Liu, Li and Liu, Qingshan},
  journal={arXiv preprint arXiv:2312.08890},
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

@article{zhu2023vdc,
  title={VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency},
  author={Zhu, Zihao and Zhang, Mingda and Wei, Shaokui and Wu, Bingzhe and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2309.16211},
  year={2023}
}

@article{zhu2023boosting,
  title={Boosting backdoor attack with a learnable poisoning sample selection strategy},
  author={Zhu, Zihao and Zhang, Mingda and Wei, Shaokui and Shen, Li and Fan, Yanbo and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2307.07328},
  year={2023}
}

@article{wang2023robust,
  title={Robust Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers},
  author={Wang, Ruotong and Chen, Hongrui and Zhu, Zihao and Liu, Li and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2306.00816},
  year={2023}
}

@article{yuan2023activation,
  title={Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks},
  author={Yuan, Danni and Wei, Shaokui and Zhang, Mingda and Liu, Li and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2312.06230},
  year={2023}
}

@article{wei2023shared,
  title={Shared adversarial unlearning: Backdoor mitigation by unlearning shared adversarial examples},
  author={Wei, Shaokui and Zhang, Mingda and Zha, Hongyuan and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2307.10562},
  year={2023}
}

@inproceedings{
    zhu2023neural,
    title={Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features},
    author={Mingli Zhu and Shaokui Wei and Hongyuan Zha and Baoyuan Wu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=VFhN15Vlkj}
}

@InProceedings{Zhu_2023_ICCV,
    author    = {Zhu, Mingli and Wei, Shaokui and Shen, Li and Fan, Yanbo and Wu, Baoyuan},
    title     = {Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4466-4477}
}

@article{liang2023badclip,
  title={BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning},
  author={Liang, Siyuan and Zhu, Mingli and Liu, Aishan and Wu, Baoyuan and Cao, Xiaochun and Chang, Ee-Chien},
  journal={arXiv preprint arXiv:2311.12075},
  year={2023}
}
```


## Copyright

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Longgang District Key Laboratory of Intelligent Digital Economy Security (iDES) at The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). iDES focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
