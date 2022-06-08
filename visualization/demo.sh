# cd BackdoorBench
# bash visualization/demo.sh
for base_model in "preactresnet18"
do
    CUDA_VISIBLE_DEVICES=7 python ./visualization/visualize.py --dataset_path ./data --dataset cifar10 --result_file_attack cifar10_${base_model}_badnet_0_1 --result_file_defense cifar10_${base_model}_badnet_0_1/anp --model ${base_model} --pratio 0.1
done