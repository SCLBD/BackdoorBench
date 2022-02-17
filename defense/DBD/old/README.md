# DBD
This a Pytorch implementation of our paper "Backdoor Defense via Disentangling
the Training Process", created by Kunzhe Huang.

**Table of Contents:**
- [DBD](#dbd)
  - [Setup](#setup)
    - [Environments](#environments)
    - [Datasets](#datasets)
    - [Log and checkpoint directories](#log-and-checkpoint-directories)
  - [Usage](#usage)
    - [No Defense](#no-defense)
    - [DBD](#dbd-1)
  - [Pretrained Models](#pretrained-models)
## Setup
### Environments
We recommend conda as the package manager to setup the environment used in our
experiments. Create the environment `dbd` from the
[environment.yml](./environment.yml) file and activate it:
```
conda env create -f environment.yml && conda activate dbd
```
**Note:** In order to speed up image transformation, we use
[pillow-simd](https://github.com/uploadcare/pillow-simd) instead of pillow to
make use of SIMD command sets with modern CPU. And Pytorch 1.6+ is required for
its native automatic mixed precision training.
### Datasets
We provide links to download datasets used in our experiments below:
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), [ImageNet-30](), [VGGFace2-30]()
Download and extract datasets to `dataset_dir` specified in the [YAML configuration files](./config).
**Note:** Make sure the path contains the sub-string `cifar`, `imagenet` or `vggface2`.

### Log and checkpoint directories
Create `saved_dir` and `storage_dir` specified in the [YAML configuration
files](./config) to save logs and checkpoints respectively:
```
mkdir saved_data && mkdir storage
``` 

## Usage
We give examples to compare standard supervised training (No Defense) and DBD on
poisoned CIFAR-10 dataset under BadNets with ResNet-18. Other settings can also
be found in the [YAML configuration files](./config), and please have an
overview before running the codes.
### No Defense
+ Run on a single GPU
    ```
    python supervise.py --config config/supervise/badnets/cifar_resnet/example.yaml \
                        --resume False \
                        --gpu 3
    ```
+ Run on multiple GPUs in distributed data parallel and turn on automatic mixed precision
    ```
    python supervise.py --config config/supervise/badnets/cifar_resnet/example.yaml \
                        --resume False \
                        --gpu "0, 1, 2, 3" \
                        --amp
    ```

### DBD
1. Self-Supervised Pre-training
   + Run on a single GPU
    ```
    python simclr.py --config config/defense/pretrain/badnets/cifar_resnet/example.yaml \
                    --resume False \
                    --gpu 3
    ```
   + Run on multiple GPUs in distributed data parallel and turn on automatic mixed precision
    ```
    python simclr.py --config config/defense/pretrain/badnets/cifar_resnet/example.yaml \
                    --resume False \
                    --gpu "0, 1, 2, 3" \
                    --amp
    ```
2. Semi-Supervised Fine-tuning

    The script [mixmatch_finetune.py](./mixmatch_finetune.py) contains the linear
    classfier training and semi-supervised fine-tuning. *It does not support
    distributed data parallel and automatic mixed precision*.
    + Run on a single GPU:
    ```
    python mixmatch_finetune.py --config config/defense/semi/badnets/cifar_resnet/example.yaml \
                                --resume False \
                                --gpu 3
    ```

## Pretrained Models
We also provide dropbox urls to download and md5sum to check pretrained models
([No Defense](checkpoint/cifar_resnet_no_defense_latest_model.txt),
[DBD](checkpoint/cifar_resnet_dbd_latest_model.txt)) in examples above. To get
the BA and ASR, please run this script [test.py](./test.py). *It does not
support distributed data parallel and automatic mixed precision*.
+ Test No Defense
  ```
  python test.py --config config/supervise/badnets/cifar_resnet/example.yaml \
                 --ckpt-dir checkpoint \
                 --resume cifar_resnet_no_defense_latest_model.pt \
                 --gpu 3
  ```

+ Test DBD
  ```
  python test.py --config config/supervise/badnets/cifar_resnet/example.yaml \
                 --ckpt-dir checkpoint \
                 --resume cifar_resnet_dbd_latest_model.pt \
                 --gpu 3
  ```