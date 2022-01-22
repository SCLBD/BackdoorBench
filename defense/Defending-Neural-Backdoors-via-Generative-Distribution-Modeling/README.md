# Defending-Neural-Backdoors-via-Generative-Distribution-Modeling
The code is for our NeurIPS 2019 paper: https://arxiv.org/abs/1910.04749

## Experiment
All the experiment we did in our paper are included in the main.py, class Experiments(). The functions are:
    
### test_contour() 
 An overall PCA result following by several figures with different beta value is plotted

### test_alpha() 
 An overall PCA result following by several figures with different alpha value is plotted

### detect_cifar() 
 Only the class been attacked exists valid triggers (You will see ASR~90%). Other classes' ASR are all very low.

### defend_cifar_10_all() 
 test defense method performance on the cifar_10 dataset

### defend_cifar_100_all() 
 test defense method performance on the cifar_100 dataset

### defend_cifar_10_all_ensemble() 
 test defense method on the cifar_10 dataset using ensemble model

### defend_cifar_100_all_ensemble() 
 test defense method on the cifar_100 dataset using ensemble model

### point_defend_cifar_10() 
 baseline method on cifar-10

### point_defend_cifar_100() 
 baseline method on cifar-100
