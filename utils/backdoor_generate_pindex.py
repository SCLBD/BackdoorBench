'''
我们测试无论是什么的准确性其实就是一个数值，表示一个映射在实验中达成的数量是多少。
限于trainer写的方式，我这边先不讨论random的情况，如果一个攻击样本我们希望output随机那就只有另写。
这里的pidx用来测试的只能是固定且唯一的。
一般用generate_pidx_from_label_transform就行
'''
import sys, logging
sys.path.append('../')
import random
import numpy as np
from typing import Callable, Union, List


def generate_single_target_attack_train_pidx(
        targets:Union[np.ndarray, List],
        tlabel: int,
        pratio: Union[float, None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
        train : bool = True,
) -> np.ndarray:
    '''
    注意！！！
    遵循bdzoo1的方式，all-to-one攻击中生成pidx不会避开target-label，
    但是测试阶段计算ASR的时候会避开target-label以免计算不准！
    '''
    logging.info('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.info('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    pidx = np.zeros(len(targets))
    if train == False:
        # Test for both clean label and normal case are the same, just skip target class samples
        # if p_num is not None or round(pratio * len(targets)):
        #     if p_num is not None:
        #         non_zero_array = np.random.choice(np.where(targets != tlabel)[0], p_num, replace = False)
        #         pidx[list(non_zero_array)] = 1
        #     else:
        #         non_zero_array = np.random.choice(np.where(targets != tlabel)[0], round(pratio * len(targets)), replace = False)
        #         pidx[list(non_zero_array)] = 1
        non_zero_array = np.where(targets != tlabel)[0]
        pidx[list(non_zero_array)] = 1
    else:
        #TRAIN !
        if clean_label == False:
            # in train state, all2one non-clean-label case NO NEED TO AVOID target class img
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.arange(len(targets)), p_num, replace = False)
                    pidx[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.arange(len(targets)), round(pratio * len(targets)), replace = False)
                    pidx[list(non_zero_array)] = 1
        else:
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], p_num, replace = False)
                    pidx[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], round(pratio * len(targets)), replace = False)
                    pidx[list(non_zero_array)] = 1
    logging.info(f'poison num:{sum(pidx)},real pratio:{sum(pidx) / len(pidx)}')
    if sum(pidx) == 0:
        raise SystemExit('No poison sample generated !')
    return pidx

def generate_multi_target_attack_train_pidx(
        targets:Union[np.ndarray, List],
        tlabel_list:List,
        pratio: Union[float, None] = None,
        p_num: Union[int,None] = None,
) -> np.ndarray:
    '''
    idea: avoid  sample with target label ( since cannot infer wheather attack succeed)

    '''
    logging.info('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.info('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    pidx = np.zeros(len(targets))
    if p_num is not None or round(pratio * len(targets)):
        if p_num is not None:
            non_zero_array = np.random.choice(np.where([True if i not in tlabel_list else False for i in targets ])[0], p_num, replace = False)
            pidx[list(non_zero_array)] = 1
        else:
            non_zero_array = np.random.choice(np.where([True if i not in tlabel_list else False for i in targets ])[0], round(pratio * len(targets)), replace = False)
            pidx[list(non_zero_array)] = 1
    # else:
    #     for (i, t) in enumerate(targets):
    #         if random.random() < pratio and t not in tlabel_list:
    #             pidx[i] = 1
    logging.info(f'poison num:{sum(pidx)},real pratio:{sum(pidx) / len(pidx)}')
    if sum(pidx) == 0:
        raise SystemExit('No poison sample generated !')
    return pidx

from utils.bd_label_transform.backdoor_label_transform import *
from typing import Optional
def generate_pidx_from_label_transform(
        original_labels: Union[np.ndarray, List],
        label_transform: Callable,
        train: bool = True,
        pratio : Union[float,None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
) -> Optional[np.ndarray]:
    '''
    idea: avoid sample with target label ( since cannot infer wheather attack succeed)
    !only support label_transform with deterministic output value (one sample one fix target label)!
    '''
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        return generate_single_target_attack_train_pidx(
            targets = original_labels,
            tlabel = label_transform.target_label,
            pratio = pratio,
            p_num = p_num,
            clean_label = clean_label,
            train = train,
        )
    #TODO only write AllToOne attack all other case need.
    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        pass
    elif isinstance(label_transform, OneToAll_randomLabelAttack):
        pass
    elif isinstance(label_transform, OneToOne_attack):
        pass
    else:
        logging.info('Not valid label_transform')



    # logging.info(f'Reminder: generate_pidx_from_label_transform only support attack that one sample has one fix target label')
    # logging.info('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    # logging.info('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    # pidx = np.zeros_like(original_labels)
    # original_labels = np.array(original_labels)
    # labels_after_transform = np.array( [label_transform(label) for label in original_labels] )
    # label_change_idx = np.where(original_labels!=labels_after_transform)[0]
    # if not is_train:
    #     logging.info('pratio does not apply during test phase')
    #     pidx[list(label_change_idx)] = 1
    # else:
    #     if p_num is not None or round(pratio * len(original_labels)):
    #         if p_num is not None:
    #             non_zero_array = np.random.choice(label_change_idx, p_num, replace = False)
    #             pidx[list(non_zero_array)] = 1
    #         else:
    #             non_zero_array = np.random.choice(label_change_idx,
    #                                     round(pratio * len(original_labels)), replace = False)
    #             pidx[list(non_zero_array)] = 1
    #     # else:
    #     #     for idx in label_change_idx:
    #     #         if random.random() < pratio:
    #     #             pidx[idx] = 1
    # logging.info(f'poison num:{sum(pidx)},real pratio:{sum(pidx) / len(pidx)}')
    # if sum(pidx) == 0:
    #     raise SystemExit('No poison sample generated !')
    # return pidx

if __name__ == '__main__':
    from bd_label_transform.backdoor_label_transform import OneToOne_attack
    label = np.array([1,2,3,2,2])
    label_transform = OneToOne_attack(1,3)
    print(generate_single_target_attack_train_pidx(label, 1, 1, 1, train = True),)
    print(generate_single_target_attack_train_pidx(label, 1, 1, 1, train = True), )
    print(generate_multi_target_attack_train_pidx(label, [1,5,6], 1, 1))
    print(generate_pidx_from_label_transform(label, AllToOne_attack(2), True, 0.5, None, ))
    print(generate_pidx_from_label_transform(label, AllToOne_attack(2), False, 0.5, None, ))