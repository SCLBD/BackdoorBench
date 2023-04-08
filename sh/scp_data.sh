# sed -i 's/\r//' ./sh/scp_data.sh
#!/bin/bash

# Example Usage
# cd bdzoo2
# bash sh/scp_data.sh

source_dir='10.20.12.241:/workspace/public_data/'
target_dir='./data/'

echo "Input your username below to access" $source_dir
read username

# Uncomment the following line to use username in this script
# username='xxxxxx'

# For CIFAR10, CIFAR100 and TinyImageNet, only zip files are needed since torchvision will unzip them.
# For GTSRB, the whole folder is needed to avoid the unzip step.

###### CIFAR10 #######
echo 'scp cifar10. Press Ctrl+C to skip.'
scp -r $username@$source_dir'cifar10' $target_dir

###### CIFAR100 ######
echo 'scp cifar100. Press Ctrl+C to skip.'
scp -r $username@$source_dir'cifar100' $target_dir

###### TINYIMAGENET ######
echo 'scp tinyimagenet. Press Ctrl+C to skip.'
scp -r $username@$source_dir'tiny' $target_dir

###### GTSRB #######
echo 'scp gtsrb. Press Ctrl+C to skip.'
scp -r $username@$source_dir'gtsrb' $target_dir

