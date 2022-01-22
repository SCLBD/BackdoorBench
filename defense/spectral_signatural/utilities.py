from collections import namedtuple
import json
import os

import numpy as np

def get_config(config_path):
    with open(config_path) as config_file:
        base_config = json.load(config_file)

    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as param_config_file:
            param_config = json.load(param_config_file)
    else:
        param_config = {}

    config = base_config
    for section, d in param_config.items():
        for k in d:
            assert k in config[section]
        config[section].update(d)
    return config


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value) 
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj
