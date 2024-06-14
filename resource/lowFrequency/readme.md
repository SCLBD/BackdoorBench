You may use code here to generate low frequency pattern for your own data.

eg.
```
python generate_pattern.py --dataset cifar10 --model vgg19_bn --clean_model_path ../../resource/clean_model/cifar10_vgg19_bn/clean_model.pth --save_folder_name lf_mask_cifar10_vgg19_bn
```

Notice that if the dataset is changed, you should change the training schedule by replace the '--yaml_path' with corresponding YAML file. (We provided YAML file for cifar10,cifar100,gtsrb,tiny in this folder.)

After running the srcipt, you can then find the npy file in the result folder. To run low frequency attack with your own pattern, you need to specify the pattern path for attack/lf.py, which is '--lowFrequencyPatternPath'.