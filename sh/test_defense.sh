cd ..

rm -rf record/test_defense_badnet_attack_1epoch

python -u attack/badnet_attack.py \
  --yaml_path ../config/attack/badnet/cifar10.yaml \
  --model preactresnet18 --pratio  0.001 \
  --save_folder_name test_defense_badnet_attack_1epoch  \
  --dataset cifar10 \
  --epochs 1

#python -u attack/inputaware_attack.py \
#  --yaml_path ../config/attack/inputaware/cifar10.yaml \
#  --model preactresnet18 --pratio  0.001 \
#  --save_folder_name test_cifar10_preactresnet18_inputaware_0_001  \
#  --dataset cifar10 \
#  --epochs 1

python -u defense/ac/ac.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10 \
  --yaml_path config/defense/ac/cifar10.yaml \
  --epochs 1

python -u defense/abl/abl.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10 \
  --yaml_path config/defense/abl/cifar10.yaml \
  --tuning_epochs 1 \
  --finetuning_epochs 1 \
  --unlearning_epochs 1

python -u defense/ft/ft.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10  \
  --yaml_path config/defense/ft/cifar10.yaml \
  --index config/defense/index/cifar10_index.txt \
  --epochs 1

python -u defense/nad/nad.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch \
  --dataset cifar10  \
  --yaml_path config/defense/nad/cifar10.yaml \
  --index config/defense/index/cifar10_index.txt \
  --epochs 1 \
  --te_epochs 1

python -u defense/spectral/spectral.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10 \
  --yaml_path config/defense/spectral/cifar10.yaml \
  --epochs 1

python -u defense/anp/anp.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10  \
  --yaml_path config/defense/anp/cifar10.yaml \
  --index config/defense/index/cifar10_index.txt \
  --epochs 1

CUDA_VISIBLE_DEVICES=0 python -u defense/dbd/dbd.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10 \
  --yaml_path config/defense/dbd/cifar10.yaml \
  --epochs 1 \
  --epoch_warmup 1 \
  --epoch_self 1


python -u defense/nc/nc.py \
  --model preactresnet18 \
  --result_file test_defense_badnet_attack_1epoch  \
  --dataset cifar10  \
  --yaml_path config/defense/nc/cifar10.yaml \
  --index config/defense/index/cifar10_index.txt \
  --epochs 1 \
  --nc_epoch 1

