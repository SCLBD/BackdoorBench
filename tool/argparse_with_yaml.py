import sys, argparse, yaml
import  logging

# print(sys.argv[0])  # print which file is the main script
sys.path.append('../')

def remove_None_from_dict(given_dict : dict) -> dict:
    return {k: v for k, v in given_dict.items() if v is not None}

# this part is load the yaml file if exists

def load_yamls_into_args(args):

    yml_config = {}

    if args.yaml_path is not None:
        try:
            with open(args.yaml_path, "r") as f:
                yml_config = yaml.safe_load(f)
        except:
            raise ValueError("Not valid yaml_path")

        if args.yaml_setting_name is None:
            pass
        else:
            # so the yaml file is a dictionary of settings, not just a dictionary of one setting
            try:
                yml_config = yml_config[args.yaml_setting_name]
            except:
                raise ValueError("Not valid setting names, not such key in yaml file")

    yml_config = remove_None_from_dict(yml_config)

    if args.additional_yaml_path is not None:
        try:
            with open(args.additional_yaml_path, "r") as f:
                addtional_yml_config = yaml.safe_load(f)
        except:
            raise ValueError("Not valid yaml_path")
        if args.additional_yaml_blocks_names is None:
            raise ValueError("No additional yaml blocks given !")
        else:
            for name in args.additional_yaml_blocks_names:
                try:
                    additional_yaml_block = remove_None_from_dict(addtional_yml_config[name])
                    for k, v in additional_yaml_block.items():
                        if k in yml_config:
                            logging.info(f"loading, additional_blocks. For {k}:{yml_config[k]} -> {k}:{v}")
                    yml_config.update(additional_yaml_block)
                except:
                    raise ValueError("Not valid setting names, not such key in yaml file")


    args.__dict__ = remove_None_from_dict(args.__dict__) # eliminate None

    for k, v in args.__dict__.items():
        if k in yml_config:
            print(f"loading command line inputs. For {k}:{yml_config[k]} -> {k}:{v}")

    yml_config.update(args.__dict__)

    yml_config = remove_None_from_dict(yml_config) # remove all key with values be None.

    args.__dict__ = yml_config

    return args

