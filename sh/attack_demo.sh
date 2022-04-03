#Here is a script of running badnets attack on cifar-10

# You should specify both the attack method script in ../attack and the YAML config file to use different attack methods.
# The detailed descriptions for each attack may be put into the `add_args` function in each script.

cd ../attack
python basicAttack.py --yaml_path ../config/basicAttack/default_badnet.yaml

# After attack you will get a folder with all files saved in ./record/xxxx (in the same folder)
# include attack_result.pt for attack model backdoored data.


