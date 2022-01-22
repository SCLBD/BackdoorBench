# Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs

<p align="center">
  <img width="750" src=https://github.com/UMBCvision/Universal-Litmus-Patterns/blob/master/docs/assets/images/teaser.png>
</p>

### Abstract
The unprecedented success of deep neural networks in many applications has made these networks a prime target for adversarial exploitation. In this paper, we introduce
a benchmark technique for detecting backdoor attacks (aka Trojan attacks) on deep convolutional neural networks (CNNs). We introduce the concept of Universal Litmus Patterns
(ULPs), which enable one to reveal backdoor attacks by feeding these universal patterns to the network and analyzing the output (i.e., classifying the network as ‘clean’
or ‘corrupted’). This detection is fast because it requires only a few forward passes through a CNN. We demonstrate the effectiveness of ULPs for detecting backdoor attacks on
thousands of networks with different architectures trained on four benchmark datasets, namely the German Traffic Sign Recognition Benchmark (GTSRB), MNIST, CIFAR10,
and Tiny-ImageNet.

### Paper
[https://arxiv.org/abs/1906.10842][paper]

### Requirements
The code was tested using pytorch 1.4.0, python 3.7.

### CIFAR-10

#### Generate poisoned data

To generate the poisoned data to be used in the experiments run
```python
python generate_poison.py
```

This script adds the triggers from ./Data/Masks to the images to generate poisoned data. Please generate one set of images for each poisoned model you want to train.
We use poisoned models for 10 triggers for training ULPs and poisoned models for the other 10 to test. 
This ensures that the train and test poisoned models use a different set of triggers.

We have made the poisoned data generated for our paper available along with the models.

#### Train models

We use a modified VGG architecture for our experiments.
To train clean models use
```python
python train_clean_model.py <partition-num> <logfile>
```

To train poisoned models use
```python
python train_poisoned_model.py <partition-num> <logfile>
```

For training ULPs: Train 500 clean models and 500 poisoned models.
For evaluating ULPs: Train 100 clean models and 100 poisoned models.

Currently each partition trains 100 models. Modify this according to your needs if you have multiple GPUs to train in parallel.

To save time, you can also use our trained models available [here](https://drive.google.com/drive/folders/1ye2KCRfzhbjtV3TMSRR5vlSBlvqNUqYL?usp=sharing):
+ extract clean_models_trainval.zip and save in ./clean_models/trainval
+ extract poisoned_models_trainval.zip and save in ./poisoned_models/trainval
+ extract clean_models_test.zip and save in ./clean_models/test
+ extract poisoned_models_test.zip and save in ./poisoned_models/test

#### Train ULPs

Once the models are generated, run
```python
python train_ULP.py <num_ULPs> <logfile> 
```

Provide appropriate number of ULPs. We run experiments for 1, 5 and 10 patterns.
This will save the results, i.e ULPs and our classifier in ./results

#### Evaluate ULPs and Noise Patterns

To evaluate ULPs run
```python
python evaluate_ULP.py 
```
To evaluate Noise patterns run
```python
python evaluate_noise.py
```

#### Plot ROC curves

```python
python plot_ROC_curves.py
```

### Tiny-ImageNet

#### Download data

Download data from the [Tiny ImageNet Visual Recognition Challenge](tiny-imagenet.herokuapp.com)
Please replace all occurrences of <tiny-imagenet-root> with the appropriate path.

#### Data cleaning

The organization of Tiny ImageNet differs from standard ImageNet. This scripts cleans the data.
```python
python data_cleaning.py
```

#### Generate poisoned data

To generate the poisoned data to be used in the experiments run
```python
python convert_data.py
python generate_poison.py
```

The first script converts the images into a numpy array and stores them in ./data for faster generation of poisons.
The second script adds the triggers from ./triggers to the images to generate poisoned data. Please generate one set of images for each poisoned model you want to train.
We use poisoned models for Triggers 01-10 for training ULPs and poisoned models for Triggers 11-20 to test. 
This ensures that the train and test poisoned models use a different set of triggers.

We have made the poisoned data generated for our paper available along with the models.

#### Train models

We use a modified Resnet architecture for our experiments.
To train clean models use
```python
python train_clean_model.py <partition-num> <logfile>
```

To train poisoned models use
```python
python train_poisoned_model.py <partition-num> <logfile>
```

For training ULPs: Train 1000 clean models and 1000 poisoned models on triggers 01-10.
For testing ULPs: Train 100 clean models and 100 poisoned models on triggers 11-20.

Currently each partition trains 50 models. Modify this according to your needs if you have multiple GPUs to train in parallel.

To save time, you can also use our trained models available [here](https://drive.google.com/drive/folders/1shYf6mUn81p0ve1DQBFhxjE_B9JN1yKt?usp=sharing):
+ extract Clean models train and save in ./clean_models/train
+ extract Poisoned models train and save in ./poisoned_models/Triggers_01_10
+ extract Clean models val and save in ./clean_models/val
+ extract Poisoned models val and save in ./poisoned_models/Triggers_11_20

#### Train ULPs

Once the models are generated, run
```python
python train_ULP.py <num_ULPs> <logfile> 
```

Provide appropriate number of ULPs. We run experiments for 1, 5 and 10 patterns.
This will save the results, i.e ULPs and our classifier in ./results

#### Evaluate ULPs and Noise Patterns

To evaluate ULPs run
```python
python evaluate_ULP.py 
```
To evaluate Noise patterns run
```python
python evaluate_noise.py
```

##### Plot ROC curves

```python
python plot_ROC_curves.py
```
### Citation
If you find our paper, code or models useful, please cite us using
```bib
@article{kolouri2019universal,
  title={Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs},
  author={Kolouri, Soheil and Saha, Aniruddha and Pirsiavash, Hamed and Hoffmann, Heiko},
  journal={arXiv preprint arXiv:1906.10842},
  year={2019}
}
```

### Acknowledgement
This work was performed under the following financial assistance award: 60NANB18D279 from U.S. Department of Commerce, National Institute of Standards and Technology, funding from SAP SE, and also NSF grant 1845216.

### Questions/Issues
Please create an issue on the Github Repo directly or contact [anisaha1@umbc.edu][email] for any questions about the code.

[paper]: https://arxiv.org/abs/1906.10842
[email]: mailto:anisaha1@umbc.edu
