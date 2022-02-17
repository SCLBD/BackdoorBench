# K-Arm Backdoor Optimization


<img src="K_Arm_Arch.png" width="600px"/>



This is the official repository of the ICML2021 paper [Backdoor Scanning for Deep Neural Networks through K-Arm Optimization](https://arxiv.org/abs/2102.05123) in PyTorch.

If you find this code is useful for your research, please cite the following:

```
@article{shen2021backdoor,
  title={Backdoor Scanning for Deep Neural Networks through K-Arm Optimization},
  author={Shen, Guangyu and Liu, Yingqi and Tao, Guanhong and An, Shengwei and Xu, Qiuling and Cheng, Siyuan and Ma, Shiqing and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2102.05123},
  year={2021}
}
```

## Setup Environments
We suggest to use Conda for testing the code on [TrojAI](https://pages.nist.gov/trojai/) datasets. Detailed instruction can be found [here](https://github.com/usnistgov/trojai-example).

### Install Anaconda Python 

[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

### Setup the Conda Environment

1. `conda create --name trojai-example python=3.8 -y`
2. `conda activate trojai-example`
3. Install required packages into this conda environment

    1. `conda install pytorch=1.7.0 torchvision=0.8.0 torchtext==0.8.0 cudatoolkit=11.0 -c pytorch -c conda-forge` 
    2. `pip install --upgrade trojai`
    3. `conda install jsonpickle`

## Datasets
### TrojAI 
The code is tested on TrojAI datasets (round1-4). TrojAI datasets can be accessed at [TrojAI website](https://pages.nist.gov/trojai/docs/data.html)
### ImageNet
We will release the ImageNet pre-trained models and code in the near future.

## Quick Start

To test the code on TrojAI datasets, simply run command

```bash
$ python main.py --result_filepath <resultFilepath> --examples_dirpath <dataDirpath> --model_filepath <modelFilepath>
```
To run the code on custom models, make sure your sample images in the `<examples_dirpath>` have the following format:
```bash
examples_dirpath/class_<class_id>_example_<example_id>.png
```


Description about the main parameters:
- `<Beta>`: Coefficient in the K-Arm schedulor objective function
- `<gamma,global(local)_theta>`: Parameters in the Arm Pre-screening procedure  
- `<global(local,ratio)_det_bound>`: Trigger size bound for detecting different types of backdoors
- `<epsilon_for_bandits>`: Controls the randomness during the K-Arm optimization


## Results
Please check our results [here](https://pages.nist.gov/trojai/docs/results.html#previous-leaderboards) under the team name `Perspecta-PurdueRutgers`. The default settings of the parameters in this repo can achieve 90% detection accuracy on TrojAI round3 training and testing datasets. For scanning different types of models, some parameters might need tunning.


## Contacts 

Guangyu Shen, [shen447@purdue.edu](shen447@purdue.edu)  
Yingqi Liu, [liu1751@purdue.edu](liu1751@purdue.edu)

