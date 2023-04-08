# BackdoorBench: a comprehensive benchmark of backdoor attack and defense methods

![Python 3.6](https://img.shields.io/badge/python-3.7-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-DodgerBlue.svg?style=plastic)
![opencv 4.5.4.60](https://img.shields.io/badge/opencv-4.5.4.60-DodgerBlue.svg?style=plastic)


### ❗ This is v1 of BackdoorBench. For v2, please refer to [link](https://github.com/SCLBD/BackdoorBench). In our new version, we provide more backdoor attack and defense methods, and more detailed documentation. 

### ❗ Important update: We have corrected the code for Label Consistent Attack in v1 also now. 

<!---
## [Overview](#overview)

<a href="#top">[Back to top]</a>
-->

BackdoorBench is a comprehensive benchmark of backdoor learning, which studies the adversarial vulnerablity of deep learning models in the training stage. It aims to provide **easy implementations** of mainstream backdoor attack and defense methods. Currently, we support:

- **Methods**
  - 8 Backdoor attack methods: [BadNets](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j), [Blended](https://arxiv.org/abs/1712.05526v1), [Label Consistent](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjvwKTx2bH4AhXCD0QIHVMWApkQFnoECAsQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1912.02771&usg=AOvVaw0NbPR9lguGTsEn3ZWtPBDR), [SIG](https://ieeexplore.ieee.org/document/8802997), [Low Frequency](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf),  [SSBA](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf), [Input-aware](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf),  [WaNet](https://openreview.net/pdf?id=eEn8KTtJOx) 
  - 9 Backdoor defense methods: FT, [Spectral](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf), [AC](http://ceur-ws.org/Vol-2301/paper_18.pdf), [FP](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13), [ABL](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf), [NAD](https://openreview.net/pdf?id=9l0K4OM-oXE), [NC](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), [DBD](https://arxiv.org/pdf/2202.03423.pdf), [ANP](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)
- **Datasets**: CIFAR-10, CIFAR100, GTSRB, Tiny ImageNet 
<!--- `mnist, cifar10, cifar100, gtsrb, celeba, tiny, imagenet`
(MNIST, CIFAR10, CIFAR100 using the pytorch official implementation, download when it is first executed. (TinyImageNet use third-party implementation, and it will be download when first executed.) The download script for GTSRB is in `./sh`. For CelebA and ImageNet, you need to download by yourself and change the dataset path argument. ) -->
- **Models**: PreAct-Resnet18, VGG19, DenseNet-161, MobileNetV3-Large, EfficientNet-B3,
<!--- `resnet18, preactresnet18, resnet34, alexnet, vgg16, vgg19, squeezenet1_0, densenet161, inception_v3, googlenet, shufflenet_v2_x1_0, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet1_0` -->

For detailed structure and implementation details, you can refer to [detailed_structure.md](./docs/detailed_structure.md).

<!--- Note that, instead of implementing each individual method separately, we try to unify the workflow of different methods, by extracting some shared modules. Consequently, it can not only ensure fair implementations of different methods, but also facilitate other researchers to quickly implement their new methhods. -->

We also provide a [**public leaderboard**](https://backdoorbench.github.io/index.html) of evaluating above backdoor attacks against above backdoor defense methods.

BackdoorBench will be continuously updated to track the lastest advances of backddor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. **You are welcome to contribute your backdoor methods to BackdoorBench.**

**News:** (2022/09/17) The manusript of the current BackdoorBench has been accepted to NeurIPS 2022 Track Datasets and Benchmarks. The arxiv version can be found at [here](https://arxiv.org/abs/2206.12654).

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
Please first to make a folder for record, all experiment results with save to record folder as default.
And make folder for data to put supported datasets.
```shell
mkdir record
mkdir data
mkdir data/cifar10
mkdir data/cifar100
mkdir data/gtsrb
mkdir data/tiny
```

Please note that due to the RAM issue, you may fail training on ImageNet. For ImageNet, please refer to the `for_imagenet` folder for a low-RAM alternative.

#### [Attack](#attack)

<a href="#top">[Back to top]</a>

This is a demo script of running badnets attack on cifar-10
```
python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_0_1
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
python ./defense/ac/ac.py --result_file badnet_0_1 --yaml_path ./config/defense/ac/cifar10.yaml --dataset cifar10
```


If you want to change the defense methods and the setting for defense, you should specify both the attack method script in ../defense and the YAML config file to use different defense methods.

### [Supported attacks](#supported-attacks)

<a href="#top">[Back to top]</a>

|                  | File name                                             | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BadNets          | [badnets_attack.py](./attack/badnets_attack.py)       | [BadNets: Identifying Vulnerabilities in  the Machine Learning Model Supply Chain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir55bv0-X2AhVJIjQIHYTjAMgQFnoECCEQAQ&url=https%3A%2F%2Fmachine-learning-and-security.github.io%2Fpapers%2Fmlsec17_paper_51.pdf&usg=AOvVaw1Cu3kPaD0a4jgvwkPCX63j) IEEE Access 2019 |
| Blended          | [blended_attack.py](./attack/blended_attack.py)       | [Targeted Backdoor Attacks on Deep  Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526v1) Arxiv 2017                                                                                                                                                                                                                                     |
| Label Consistent | [lc_attack.py](./attack/lc_attack.py)                 | [Label-Consistent Backdoor Attacks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjvwKTx2bH4AhXCD0QIHVMWApkQFnoECAsQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1912.02771&usg=AOvVaw0NbPR9lguGTsEn3ZWtPBDR) Arxiv 2019                                                                                                 |
| SIG              | [sig_attack.py](./attack/sig_attack.py)               | [A new backdoor attack in  cnns by training set corruption](https://ieeexplore.ieee.org/document/8802997) ICIP 2019                                                                                                                                                                                                                                           |
| Low Frequency    | [lf_attack.py](./attack/lf_attack.py)                 | [Rethinking the Backdoor Attacks’ Triggers: A Frequency Perspective](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf) ICCV2021                                                                                                                                |
| SSBA             | [ssba_attack.py](./attack/ssba_attack.py)             | [Invisible Backdoor Attack with  Sample-Specific Triggers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                                    |
| Input-aware      | [inputaware_attack.py](./attack/inputaware_attack.py) | [Input-Aware Dynamic Backdoor Attack](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf) NeurIPS 2020                                                                                                                                                                                                                 |
| WaNet            | [wanet_attack.py](./attack/wanet_attack.py)           | [WaNet -- Imperceptible  Warping-Based Backdoor Attack](https://openreview.net/pdf?id=eEn8KTtJOx) ICLR 2021                                                                                                                                                                                                                                                   |

For SSBA, the file we used with 1-bit embedded in the images is given at https://drive.google.com/drive/folders/1QU771F2_1mKgfNQZm3OMCyegu2ONJiU2?usp=sharing .
  
For lc attack the file we used is at https://drive.google.com/drive/folders/1Qhj5vXX7kX74IWdrQDwguWsV8UvJmzF4 .
  
For lf attack the file we used is at https://drive.google.com/drive/folders/16JrANmjDtvGc3lZ_Cv4lKEODFjRebmvk .

### [Supported defenses](#supported-defsense) 

<a href="#top">[Back to top]</a>

|       | File name                 | Paper                |
| :------------- |:-------------|:-----|
| FT| [ft.py](./defense/ft/ft.py) | standard fine-tuning|
| FP | [fp.py](./defense/fp/fp.py) | [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) RAID 2018 |
| NAD      | [nad.py](./defense/nad/nad.py)                | [Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks](https://openreview.net/pdf?id=9l0K4OM-oXE) ICLR 2021 |
| NC       | [nc.py](./defense/nc/nc.py)                   | [Neural Cleanse: Identifying And Mitigating Backdoor Attacks In Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8835365), IEEE S&P 2019 |
| ANP      | [anp.py](./defense/anp/anp.py)                | [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf) NeurIPS 2021 |
| AC       | [ac.py](./defense/ac/ac.py)                   | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](http://ceur-ws.org/Vol-2301/paper_18.pdf) ceur-ws 2018 |
| Spectral | [spectral.py](./defense/spectral/spectral.py) | [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf) NeurIPS 2018 |
| ABL      | [abl.py](./defense/abl/abl.py)                | [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://proceedings.neurips.cc/paper/2021/file/7d38b1e9bd793d3f45e0e212a729a93c-Paper.pdf) NeurIPS 2021 |
| DBD | [dbd.py](./defense/dbd/dbd.py) | [Backdoor Defense Via Decoupling The Training Process](https://arxiv.org/pdf/2202.03423.pdf) ICLR 2022 |


### [Results](#results)

<a href="#top">[Back to top]</a>

We present partial results on cifar10 with poison ratio = 1% here as an example. For complete results, please refer to our [leaderboard](https://backdoorbench.github.io/leaderboard-cifar10.html).

|                   | BackdoorDefense→     | Nodefense    | Nodefense    | Nodefense    | FT        | FT      | FT        | FP        | FP      | FP        | NAD       | NAD     | NAD       | NC        | NC      | NC        | ANP       | ANP     | ANP       | AC        | AC      | AC        | Spectral  | Spectral | Spectral  | ABL       | ABL     | ABL       | DBD       | DBD     | DBD       |
| ----------------- | -------------------- | ------------ | ------------ | ------------ | --------- | ------- | --------- | --------- | ------- | --------- | --------- | ------- | --------- | --------- | ------- | --------- | --------- | ------- | --------- | --------- | ------- | --------- | --------- | -------- | --------- | --------- | ------- | --------- | --------- | ------- | --------- |
| TargetedModel     | BackdoorAttack↓     | C-Acc (%)    | ASR (%)      | R-Acc (%)    | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%)  | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) | C-Acc (%) | ASR (%) | R-Acc (%) |
| PreAct-Resnet18   | BadNets              | 91.32        | 95.03        | 4.67         | 89.96     | 1.48    | 89.39     | 91.31     | 57.13   | 41.62     | 89.87     | 2.14    | 88.71     | 89.05     | 1.27    | 89.16     | 84.17     | 0.73    | 84.04     | 88.8      | 86.23   | 13.28     | 89.98     | 92.41    | 7.2       | 83.32     | 0       | 89.02     | 89.65     | 1.28    | 89.17     |
| PreAct-Resnet18   | Blended              | 93.47        | 99.92        | 0.08         | 92.78     | 96.11   | 3.57      | 93.17     | 99.26   | 0.73      | 92.17     | 97.69   | 2.14      | 93.47     | 99.92   | 0.08      | 86.58     | 2.19    | 61.17     | 88.52     | 99.72   | 0.28      | 90.35     | 99.84    | 0.14      | 77.3      | 0.73    | 64.52     | 69.91     | 99.98   | 0.02      |
| PreAct-Resnet18   | LC                   | 84.59        | 99.92        | 0.07         | 90.6      | 11.53   | 59.74     | 89.53     | 0.46    | 13.32     | 90.42     | 10.5    | 59.7      | 90.28     | 6.87    | 62.24     | 83.98     | 4.12    | 54.69     | 81.28     | 98.78   | 0.86      | 84.13     | 99.53    | 0.42      | 65.31     | 0       | 52.69     | 61.71     | 0       | 54.01     |
| PreAct-Resnet18   | SIG                  | 84.48        | 98.27        | 1.72         | 90.8      | 2.37    | 69.74     | 89.1      | 26.2    | 20.61     | 90.02     | 10.66   | 64.2      | 84.48     | 98.27   | 1.72      | 77.97     | 0.01    | 48.8      | 82.41     | 94.61   | 5.17      | 83.01     | 92.27    | 7.4       | 57.8      | 0       | 59.97     | 60.67     | 100     | 0         |
| PreAct-Resnet18   | LF                   | 93.19        | 99.28        | 0.71         | 92.37     | 78.44   | 19.42     | 92.9      | 98.97   | 1.02      | 92.37     | 47.83   | 47.49     | 91.62     | 1.41    | 87.48     | 85.15     | 0.6     | 82.72     | 88.92     | 98.09   | 1.72      | 92.58     | 98.67    | 1.28      | 81.72     | 0.21    | 84.81     | 50.98     | 99.7    | 0.08      |
| PreAct-Resnet18   | SSBA                 | 92.88        | 97.86        | 1.99         | 92.14     | 74.79   | 23.31     | 92.54     | 83.5    | 15.36     | 91.91     | 77.4    | 20.86     | 90.99     | 0.58    | 87.04     | 86.54     | 0.39    | 79.87     | 90        | 96.23   | 3.53      | 89.63     | 90.5     | 8.71      | 80.79     | 0       | 84.66     | 63.5      | 99.51   | 0.39      |
| PreAct-Resnet18   | Input-aware          | 90.67        | 98.26        | 1.66         | 93.12     | 1.72    | 90.53     | 91.74     | 0.04    | 44.54     | 93.18     | 1.68    | 91.12     | 92.61     | 0.76    | 90.87     | 90.36     | 1.64    | 86.99     | 91.48     | 88.62   | 10.61     | 91.39     | 90.43    | 9.13      | 63.19     | 90.16   | 5.77      | 78.6      | 8.54    | 75.28     |
| PreAct-Resnet18   | WaNet                | 91.25        | 89.73        | 9.76         | 93.48     | 17.1    | 78.29     | 91.46     | 1.09    | 69.73     | 93.17     | 22.98   | 72.69     | 91.8      | 7.53    | 85.09     | 85.73     | 0.43    | 86.5      | 91.93     | 96.8    | 3.06      | 91.94     | 90.17    | 9.37      | 83.19     | 0       | 85.86     | 80.9      | 6.61    | 77.61     |
| VGG19             | BadNets              | 89.36        | 95.93        | 3.81         | 88.08     | 25.42   | 69.68     | 89.23     | 92.61   | 6.82      | 87.51     | 38.17   | 58.3      | 87.86     | 1       | 88.01     | NA        | NA      | NA        | 86.25     | 94.37   | 5.17      | 88.07     | 91.67    | 7.78      | 85.13     | 93.92   | 5.5       | 10        | 100     | 0         |
| VGG19             | Blended              | 90.17        | 99.12        | 0.82         | 88.76     | 94.37   | 5.22      | 90.07     | 99.11   | 0.82      | 88.35     | 93.08   | 6.33      | 85.92     | 1.79    | 74.13     | NA        | NA      | NA        | 87.58     | 98.13   | 1.76      | 89.18     | 99.76    | 0.23      | 88.61     | 99.48   | 0.46      | 10        | 100     | 0         |
| VGG19             | LC                   | 82.21        | 97.46        | 1.78         | 85.85     | 7.41    | 56.07     | 85.35     | 96.84   | 2.29      | 85.34     | 6.43    | 56.57     | 82.21     | 97.46   | 1.78      | NA        | NA      | NA        | 77.89     | 35.71   | 39.92     | 80.65     | 73.09    | 18.9      | 80.73     | 98.11   | 1.21      | 10        | 0       | 11.1      |
| VGG19             | SIG                  | 81.69        | 99.8         | 0.12         | 86.89     | 98.16   | 1.68      | 84.52     | 99.93   | 0.07      | 86.01     | 99.18   | 0.77      | 81.69     | 99.8    | 0.12      | NA        | NA      | NA        | 79.6      | 99.94   | 0.06      | 80.27     | 97.77    | 1.89      | 80.53     | 99.81   | 0.16      | 10        | 0       | 11.11     |
| VGG19             | LF                   | 88.94        | 93.93        | 5.62         | 87.88     | 62.84   | 33.01     | 88.98     | 91.8    | 7.46      | 87.85     | 59.38   | 35.01     | 85.35     | 9.99    | 72.79     | NA        | NA      | NA        | 83.07     | 89.39   | 8.59      | 87.88     | 92.83    | 6.44      | 87.89     | 93.17   | 6         | 10        | 100     | 0         |
| VGG19             | SSBA                 | 89.48        | 91.86        | 7.29         | 88.2      | 46.28   | 47.83     | 89.4      | 89.66   | 9.22      | 87.65     | 37.54   | 54.58     | 89.48     | 91.86   | 7.29      | NA        | NA      | NA        | 88.62     | 92.26   | 6.98      | 89.61     | 92.6     | 6.81      | 87.66     | 8.69    | 78.41     | 10        | 0       | 11.11     |
| VGG19             | Input-aware          | 77.69        | 94.59        | 4.79         | 74.74     | 13.39   | 57.97     | 78.62     | 86.77   | 11.79     | 75.7      | 23.36   | 54.71     | 77.67     | 94.58   | 4.79      | NA        | NA      | NA        | 83.52     | 82.57   | 15.2      | 87.22     | 84.71    | 13.58     | 88.3      | 86.48   | 12        | 10        | 100     | 0         |
| VGG19             | WaNet                | 88.43        | 88.9         | 10.3         | 91.03     | 30.04   | 64.93     | 89.61     | 73.39   | 24.57     | 90.82     | 44.93   | 51.18     | 88.43     | 88.89   | 10.3      | NA        | NA      | NA        | 87.96     | 92.56   | 7.02      | 87.76     | 96.24    | 3.53      | 86.29     | 91.77   | 7.22      | 10        | 100     | 0         |
| EfficientNet-B3   | BadNets              | 57.48        | 9.51         | 55.59        | 53.69     | 5.09    | 52.96     | 57.97     | 3.88    | 57.46     | 55.29     | 4.5     | 54.68     | 57.48     | 9.53    | 55.58     | 57.48     | 9.53    | 55.58     | 39.76     | 17.28   | 36.51     | 37.64     | 15.5     | 36.11     | 36.44     | 8.51    | 36.29     | 54.44     | 94.26   | 3.91      |
| EfficientNet-B3   | Blended              | 65.11        | 87.76        | 8.01         | 61.16     | 8.6     | 44.66     | 65.21     | 59.03   | 24.14     | 62.76     | 11.86   | 46.32     | 59.93     | 3.87    | 51.8      | 65.11     | 87.76   | 8.01      | 34.58     | 63.32   | 15.88     | 53.68     | 70.81    | 17.58     | 18.14     | 1.91    | 15.51     | 52.1      | 99.94   | 0.04      |
| EfficientNet-B3   | LC                   | 62.09        | 1.61         | 47.72        | 60.68     | 7.34    | 42.04     | 64.07     | 7.02    | 44.89     | 62.51     | 7.22    | 42.8      | 57.32     | 7.1     | 41.63     | 62.09     | 1.61    | 47.73     | 26.55     | 0       | 28.11     | 45.53     | 0.04     | 42.14     | 22.42     | 0.03    | 23.32     | 60.13     | 0       | 53.02     |
| EfficientNet-B3   | SIG                  | 58.14        | 99.71        | 0.13         | 56.92     | 3.98    | 24.03     | 58.83     | 0.12    | 10.84     | 58.27     | 3.63    | 25.83     | 58.15     | 99.71   | 0.13      | 58.15     | 99.71   | 0.13      | 46.07     | 99.16   | 0.26      | 52.64     | 98.14    | 0.76      | 11.79     | 9.58    | 32.02     | 45.49     | 99.96   | 0.04      |
| EfficientNet-B3   | LF                   | 56.95        | 80.06        | 10.51        | 53.91     | 26.43   | 31.87     | 56.71     | 64.11   | 18.17     | 55.09     | 34.42   | 30.13     | 52.09     | 16.62   | 37.49     | 53.01     | 67.98   | 15.43     | 34.42     | 72.51   | 9.41      | 52.27     | 68.96    | 15.03     | 41.86     | 32.03   | 32.39     | 50.48     | 99.21   | 0.58      |
| EfficientNet-B3   | SSBA                 | 55.76        | 24.4         | 45.5         | 53.42     | 7.1     | 49.32     | 56.49     | 6.07    | 52.97     | 54.7      | 6.91    | 50.72     | 55.77     | 24.4    | 45.49     | 55.77     | 24.4    | 45.49     | 29.94     | 29.13   | 25.88     | 49.67     | 13.8     | 45.66     | 27.37     | 15.74   | 24.19     | 51.82     | 99.27   | 0.59      |
| EfficientNet-B3   | Input-aware          | 70.01        | 94.96        | 2.34         | 67.84     | 50.37   | 17.97     | 65.61     | 7.03    | 15.66     | 70.3      | 48.93   | 22.64     | 67.47     | 54.81   | 8.14      | 70.01     | 29.92   | 27.78     | 47.2      | 99.39   | 0.07      | 42.22     | 98.13    | 0.26      | 17.3      | 16.47   | 17.13     | 54.13     | 100     | 0         |
| EfficientNet-B3   | WaNet                | 71.73        | 7.93         | 69.33        | 70.8      | 3.59    | 67.96     | 25.27     | 7.43    | 26.2      | 73.23     | 3.14    | 70.88     | 71.74     | 7.93    | 69.33     | 71.74     | 7.93    | 69.33     | 37.78     | 14.34   | 35.28     | 52.44     | 7.11     | 51.91     | 37.97     | 7.66    | 35.89     | 58.83     | 18.44   | 54.87     |
| MobileNetV3-Large | BadNets              | 82.45        | 93.39        | 5.84         | 77.99     | 3       | 77        | 82.36     | 93.97   | 5.32      | 79.02     | 4.57    | 76.22     | 75.19     | 2.53    | 74.62     | 82.04     | 3.72    | 80.87     | 70.46     | 89.98   | 7.77      | 78.12     | 92.83    | 5.93      | 76.57     | 83.59   | 13.33     | 57.21     | 86.84   | 9.67      |
| MobileNetV3-Large | Blended              | 83.43        | 98.76        | 1.13         | 78.94     | 12.27   | 53.93     | 83.13     | 1.46    | 19.52     | 79.88     | 18.81   | 52.72     | 77.59     | 3.53    | 65.72     | 75.56     | 95.02   | 3.97      | 71.48     | 95.08   | 3.33      | 79.18     | 98.16    | 1.46      | 76        | 79.02   | 15.63     | 51.67     | 99.64   | 0.31      |
| MobileNetV3-Large | LC                   | 75.53        | 98.67        | 0.84         | 76.39     | 6.47    | 49.82     | 80.78     | 0.12    | 11.78     | 78.13     | 5.92    | 50.24     | 76        | 6.64    | 48.94     | 70.21     | 69.12   | 14.1      | 65.83     | 95.94   | 2.13      | 73.47     | 98.62    | 0.86      | 66.66     | 99.98   | 0.02      | 56.71     | 0       | 49.44     |
| MobileNetV3-Large | SIG                  | 77.13        | 98.7         | 1.04         | 78.09     | 2.51    | 42.93     | 82        | 0       | 17.41     | 79.1      | 1.87    | 37.86     | 77.13     | 98.7    | 1.04      | 73.49     | 78.3    | 14.4      | 67.85     | 99.33   | 0.46      | 73.47     | 97.99    | 1.33      | 63.2      | 90.66   | 5.63      | 48.42     | 100     | 0         |
| MobileNetV3-Large | LF                   | 82.73        | 96.63        | 1.77         | 77.92     | 24.54   | 44.76     | 82.22     | 0.6     | 13.33     | 78.98     | 10.08   | 57        | 82.73     | 96.63   | 1.77      | 75.45     | 5.44    | 58.76     | 72.37     | 96.33   | 2.32      | 79.61     | 96.58    | 2.42      | 75.32     | 64.92   | 25.42     | 51.4      | 99.37   | 0.49      |
| MobileNetV3-Large | SSBA                 | 81.84        | 85.06        | 12.16        | 77.8      | 6.94    | 68.71     | 82.14     | 62.99   | 29.57     | 78.35     | 9.51    | 68.06     | 75.95     | 3.96    | 71.43     | 76.26     | 58.59   | 30.71     | 67.79     | 57.71   | 29.01     | 76.85     | 72.18    | 21.61     | 76.22     | 31.27   | 50.26     | 53.93     | 97.93   | 1.51      |
| MobileNetV3-Large | Input-aware          | 77.35        | 89.17        | 9.12         | 78.2      | 7.09    | 62.23     | 79.8      | 3.32    | 57.37     | 79.82     | 4.46    | 64.8      | 76.77     | 4.37    | 67.01     | 77.35     | 89.17   | 9.12      | 71.36     | 69.29   | 21.37     | 78.58     | 75.03    | 19.74     | 73.96     | 78.38   | 15.32     | 50.01     | 99.98   | 0.01      |
| MobileNetV3-Large | WaNet                | 78           | 70.88        | 24.8         | 80.4      | 4.14    | 77.22     | 24.71     | 9.89    | 20.82     | 81.81     | 3.56    | 78.97     | 78        | 70.88   | 24.8      | 78.57     | 4.71    | 76.3      | 69.61     | 43.82   | 41.47     | 77.59     | 53.23    | 36.92     | 59.83     | 58.71   | 29.79     | 58.66     | 12.3    | 54.99     |
| DenseNet-161      | BadNets              | 84.33        | 89.68        | 9.1          | 83.59     | 32.7    | 56.87     | 85.16     | 87.51   | 10.81     | 82.28     | 68.73   | 26.47     | 83.1      | 1.84    | 82.36     | 76.53     | 0.29    | 79.92     | 73.17     | 79.32   | 16.6      | 81.13     | 87.71    | 10.59     | 76.32     | 1.02    | 79.38     | 67.41     | 15.23   | 64.12     |
| DenseNet-161      | Blended              | 86.37        | 98.79        | 1.04         | 84.39     | 81.83   | 14.74     | 85.95     | 98.98   | 0.88      | 83.69     | 72.82   | 19.6      | 83.17     | 1.01    | 72.9      | 78.93     | 2.06    | 65.93     | 75.45     | 94.31   | 4.23      | 82.06     | 97.01    | 2.44      | 67.81     | 2.16    | 73.28     | 56.66     | 99.53   | 0.4       |
| DenseNet-161      | LC                   | 78.68        | 78.06        | 13.88        | 83.58     | 9.12    | 52.66     | 83.91     | 41.49   | 36.56     | 82.34     | 8.68    | 52.36     | 81.45     | 7.33    | 52.31     | 80.16     | 20.92   | 47.42     | 68.92     | 15.27   | 45.64     | 77.02     | 40.88    | 36.48     | 58.79     | 19.9    | 39.38     | 72.36     | 0       | 60.77     |
| DenseNet-161      | SIG                  | 78.63        | 98.67        | 1.23         | 83.49     | 19.3    | 49.21     | 83.23     | 27.57   | 38.66     | 82.7      | 18.47   | 48.69     | 81.62     | 7.99    | 43.91     | 74.25     | 2.64    | 45.37     | 69.85     | 98.34   | 1.33      | 75.61     | 98.71    | 1.18      | 43.47     | 0       | 60.47     | 45.4      | 96.77   | 3.1       |
| DenseNet-161      | LF                   | 84.32        | 91.7         | 6.9          | 83.07     | 62.07   | 31.2      | 83.93     | 92.04   | 6.42      | 82.51     | 62.29   | 30.13     | 81.11     | 10.73   | 67.6      | 78.5      | 6.73    | 70.53     | 72.91     | 83.24   | 12.26     | 80.61     | 88.47    | 9.2       | 64.4      | 23.01   | 53.82     | 59.62     | 98.29   | 1.46      |
| DenseNet-161      | SSBA                 | 84.18        | 84.13        | 13.39        | 83.76     | 32.84   | 55.9      | 84.26     | 83.76   | 13.49     | 82.8      | 29.79   | 57.22     | 81.68     | 8.49    | 71.78     | 76.82     | 1.23    | 74.62     | 71.94     | 60.54   | 29.57     | 79.68     | 75.13    | 20.34     | 68.13     | 0.59    | 77.33     | 67.25     | 97.62   | 1.93      |
| DenseNet-161      | Input-aware          | 84.46        | 94.41        | 5.18         | 86.77     | 2.69    | 82.02     | 85.15     | 22.34   | 60.34     | 86.47     | 10.31   | 76.64     | 84.45     | 94.41   | 5.18      | 84.03     | 4.06    | 75.08     | 74.19     | 89.66   | 7.73      | 81.64     | 91.82    | 6.68      | 63.79     | 13.14   | 51.26     | 74.12     | 77.45   | 17.14     |
| DenseNet-161      | WaNet                | 84.61        | 73.81        | 22.72        | 87.52     | 2.34    | 84.48     | 85.72     | 24.61   | 64.71     | 87.3      | 6.26    | 81.16     | 87.1      | 1.86    | 85.07     | 84.21     | 1.24    | 83.69     | 72.07     | 41.99   | 43.37     | 80.04     | 65.82    | 28.01     | 30.37     | 97.74   | 1.58      | 65.25     | 10.98   | 62        |

## [Citation](#citation)

<a href="#top">[Back to top]</a>

If interested, you can read our recent works about backdoor learning, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@inproceedings{wu2022backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao and Zha, Hongyuan},
  journal={NeurIPS 2022 Track Datasets and Benchmarks},
  year={2022}
}

@inproceedings{wu2022backdoordefense,
  title={Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples},
  author={Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
  booktitle={Neural Information Processing Systems},
  year={2022}
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

@article{gao2022imperceptible,
  title={Imperceptible and Robust Backdoor Attack in 3D Point Cloud},
  author={Gao, Kuofeng and Bai, Jiawang and Wu, Baoyuan and Ya, Mengxi and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2208.08052},
  year={2022}
}
```


## [Copyright](#copyright)

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) and [Shenzhen Research Institute of Big Data](http://www.sribd.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If any suggestion or comment, please contact us at <wubaoyuan@cuhk.edu.cn>.
