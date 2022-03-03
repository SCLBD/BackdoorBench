import logging
import re

class translate_layer_name_for_eval_class(object):

    def __init__(self):
        self.warn_list = []

    def __call__(self, layer_name):
        new_layer_name = translate_layer_name_for_eval(layer_name)
        if layer_name not in self.warn_list:
            if layer_name != new_layer_name:
                logging.warning(
                    f'find layer_name in named_modules() format, and transform {layer_name} to {new_layer_name}')
                self.warn_list.append(layer_name)
        return new_layer_name

def translate_layer_name_for_eval(layer_name):
    # do layername transform in order to pass eval
    # eg "layer4.1.bn2"to "layer4[1].bn2 "

    old_layer_name = layer_name
    new_layer_name = re.sub(r'\.(\d)(\.|$)', r'[\1]\2', old_layer_name)
    return new_layer_name