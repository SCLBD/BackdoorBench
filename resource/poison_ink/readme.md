This folder is to generate poison data in poison_ink.py ("--attack_train_replace_imgs_path" and 
"--attack_test_replace_imgs_path" should receive two path for poisoned training data and poisoned testing data, respectively).

Step 1: 

Choose the dataset you want to poison, and run the following command to convert the dataset into seperate image files in the same folder:
    eg. 

```shell
python dataset_convert_into_images.py --dataset {dataset_name}
```

[//]: # (```shell)

[//]: # (python dataset_convert_into_images.py --dataset cifar10)

[//]: # (```)

(If you do not want to use the dataset we provided, you can also use your own dataset. Then you should put the training images and testing images separately into two folders. These folder should have structure like this: `./{train|test}/{class_index}/{image_index}.png`



Step 2:

Run the following command to train the autoencoder on given **training** dataest:

**!!! Important Note !!!**
I use the script for training(Our rewritten verison only changes the protocols), and during our experiment, we found if we follow the 
instruction in the paper (4 stage, incremental training: rot -> crop -> flip -> adv), the model CANNOT generate image with high ASR 
(Our experimen on cifar10 with ASR~50%, we guess it can be hyperparameter issue, since we found loss grows up with default hyperparameters).
Since by now, the author has not response to this issue (both on e-mail and git issue), we only use first 3 stages! (rot -> crop -> flip, **NO** final adv stage!).
And once we get the response from the author, we may update the code and the instruction. 

eg.

```shell
python train.py --datadir ../../data/cifar10_seperate_images --ngpu 1 --remark cifar10_rot --rot
python train.py --datadir ../../data/cifar10_seperate_images --ngpu 1 --remark cifar10_crop --crop \
    --Hnet ./chk/cifar10_rot/checkPoints/netH_epoch_199.pth \
    --Rnet ./chk/cifar10_rot/checkPoints/netR_epoch_199.pth \
    --Dnet ./chk/cifar10_rot/checkPoints/netD_epoch_199.pth
python train.py --datadir ../../data/cifar10_seperate_images --ngpu 1 --remark cifar10_flip --flip \
    --Hnet ./chk/cifar10_crop/checkPoints/netH_epoch_199.pth \
    --Rnet ./chk/cifar10_crop/checkPoints/netR_epoch_199.pth \
    --Dnet ./chk/cifar10_crop/checkPoints/netD_epoch_199.pth
#    python train.py --datadir ../../data/cifar10_seperate_images --ngpu 1 --remark cifar10_adv --adv \
#        --Hnet ./chk/cifar10_flip/checkPoints/netH_epoch_199.pth \
#        --Rnet ./chk/cifar10_flip/checkPoints/netR_epoch_199.pth \
#        --Dnet ./chk/cifar10_flip/checkPoints/netD_epoch_199.pth
  
```

[//]: # (```shell)

[//]: # (python train.py --datadir ../../data/cifar10_seperate_images --remark cifar10 --niter 1)

[//]: # (  rm -rf ./chk/)

[//]: # (```)
  Then you can find checkpoint of network H at `./chk/{exp_name}/checkPoints/`

[//]: # (    `./chk/cifar10/checkPoints/netH_epoch_0.pth`)
Step 3:

For trigger generation, note that our implementation is different from the original one in 2 part:
1. we generate all samples' poisoned version (for random sample and poison ratio, we can do this in the training stage of the model)
2. we ignore some parts in the code which are not mentioned in the paper. (All those parts which cannot be guessed from both the abspath and paper, 
eg. `advdir = "/public/zhangjie/ECCV_2020/backdoor/data_processing_zk/data/cifar10_vgg19_totensor_norm_LR0.01/adv_PGD-8/test`, 
Since by now, the author has not response to this issue (both on e-mail and git issue), we ignore these parts and only keep other parts mentioned in the paper,
once get response, we may update these parts).

Run the following command to generate the poisoned training and testing data:

eg.

```shell
python trigger_generation.py --data_dir ../../data/{dataset_name}_seperate_images --Hnet {your_trained_model_path} --train
python trigger_generation.py --data_dir ../../data/{dataset_name}_seperate_images --Hnet {your_trained_model_path} --test
```

[//]: # (```shell)

[//]: # (python trigger_generation.py --data_dir ../../data/cifar10_seperate_images --Hnet ./chk/cifar10/checkPoints/netH_epoch_0.pth --train)

[//]: # (python trigger_generation.py --data_dir ../../data/cifar10_seperate_images --Hnet ./chk/cifar10/checkPoints/netH_epoch_0.pth --test)

[//]: # (```)
Then you can find result at `./train` and `./test`, these data will be used in poison_ink.py by
`--attack_train_replace_imgs_path ./resource/poison_ink/train` and `--attack_test_replace_imgs_path ./resource/poison_ink/test` respectively.