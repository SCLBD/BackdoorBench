# sed -i 's/\r//' ./sh/scp_data.sh
#!/bin/bash

# Example Usage
# cd bdzoo2
# bash sh/scp_data.sh

source_dir='10.20.12.241:/workspace/public_resource'
target_dir='./resource'

echo "Input your username below to access" $source_dir
read username

# Uncomment the following line to use username in this script
# username='xxxxxx'

# For CIFAR10, CIFAR100 and TinyImageNet, only zip files are needed since torchvision will unzip them.
# For GTSRB, the whole folder is needed to avoid the unzip step.

echo 'scp resource'
scp -r $username@$source_dir $target_dir

