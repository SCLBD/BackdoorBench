# cd BackdoorBench
# bash visualization/demo.sh
for base_model in "preactresnet18" "vgg19" "efficientnet_b3" "mobilenet_v3_large" "densenet161"
do
    python ./visualization/visualize.py --dataset_path ./data --dataset tiny --result_file_attack tiny_${base_model}_badnet_0_05 --result_file_defense tiny_${base_model}_badnet_0_05/ac --model ${base_model}
done