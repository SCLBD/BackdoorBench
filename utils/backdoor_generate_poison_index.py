# idea: this file is for the poison sample index selection,
#   generate_single_target_attack_train_poison_index is for all-to-one attack label transform
#   generate_poison_index_from_label_transform aggregate both all-to-one and all-to-all case.

import sys, logging
sys.path.append('../')
import random
import numpy as np
from typing import Callable, Union, List


def generate_single_target_attack_train_poison_index(
        targets:Union[np.ndarray, List],
        tlabel: int,
        pratio: Union[float, None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
        train : bool = True,
) -> np.ndarray:
    '''
    # idea: given the following information, which samples will be used to poison will be determined automatically.

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    targets = np.array(targets)
    logging.debug('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.debug('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    poison_index = np.zeros(len(targets))
    if train == False:

        non_zero_array = np.where(targets != tlabel)[0]
        poison_index[list(non_zero_array)] = 1
    else:
        #TRAIN !
        if clean_label == False:
            # in train state, all2one non-clean-label case NO NEED TO AVOID target class img
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.arange(len(targets)), p_num, replace = False)
                    poison_index[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.arange(len(targets)), round(pratio * len(targets)), replace = False)
                    poison_index[list(non_zero_array)] = 1
        else:
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], p_num, replace = False)
                    poison_index[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], round(pratio * len(targets)), replace = False)
                    poison_index[list(non_zero_array)] = 1
    logging.info(f'poison num:{sum(poison_index)},real pratio:{sum(poison_index) / len(poison_index)}')
    if sum(poison_index) == 0:
        raise SystemExit('No poison sample generated !')
    return poison_index

from utils.bd_label_transform.backdoor_label_transform import *
from typing import Optional
def generate_poison_index_from_label_transform(
        original_labels: Union[np.ndarray, List],
        label_transform: Callable,
        train: bool = True,
        pratio : Union[float,None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
) -> Optional[np.ndarray]:
    '''

    # idea: aggregate all-to-one case and all-to-all cases, case being used will be determined by given label transformation automatically.

    !only support label_transform with deterministic output value (one sample one fix target label)!

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select (only in all2one case can be used !!!)
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    if clean_label:
        logging.warning("clean_label = True! Note that in our implementation poisoning ratio is ALWAYS defined as number of poisoning samples / number of all samples.")
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        return generate_single_target_attack_train_poison_index(
            targets = original_labels,
            tlabel = label_transform.target_label,
            pratio = pratio,
            p_num = p_num,
            clean_label = clean_label,
            train = train,
        )

    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        if train:
            pass
        else:
            p_num = None
            pratio = 1

        if p_num is not None:
            select_position = np.random.choice(len(original_labels), size = p_num, replace=False)
        elif pratio is not None:
            select_position = np.random.choice(len(original_labels), size=round(len(original_labels) * pratio), replace=False)
        else:
            raise SystemExit('p_num or pratio must be given')
        logging.info(f'poison num:{len(select_position)},real pratio:{len(select_position) / len(original_labels)}')

        poison_index = np.zeros(len(original_labels))
        poison_index[select_position] = 1

        return poison_index
    else:
        logging.debug('Not valid label_transform')



