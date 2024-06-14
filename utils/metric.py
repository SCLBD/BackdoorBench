'''Define some commonly used metric for evaluation'''

import numpy as np
import torch


'''clean accuracy (C-Acc) (i.e., the prediction accuracy of clean samples),'''
def clean_accuracy(pred, label):
    '''Compute the accuracy of clean samples'''
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(label, list):
        label = np.array(label)
        
    return np.mean((pred == label))

def clean_accuracy_per_class(pred, label, num_classes):
    '''Compute the accuracy of clean samples per class'''
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(label, list):
        label = np.array(label)
        
    accuracy = []
    for i in range(num_classes):
        accuracy.append(np.mean((pred[label == i] == i)))
    return accuracy


'''attack success rate (ASR) (i.e., the prediction accuracy of poisoned samples to the target class)'''
def attack_success_rate(pred, target_label):
    '''Compute the attack success rate'''

    return clean_accuracy(pred, target_label)

def attack_success_rate_per_class(pred, target_label, num_classes):
    '''Compute the attack success rate per class'''
    return clean_accuracy_per_class(pred, target_label, num_classes)



'''robust accuracy (R-Acc) (i.e., the prediction accuracy of poisoned samples to the original class)'''
def robust_accuracy(bd_pred, ori_label):
    '''Compute the robust accuracy'''
    return clean_accuracy(bd_pred, ori_label)

def robust_accuracy_per_class(bd_pred, ori_label, num_classes):
    '''Compute the robust accuracy per class'''
    return clean_accuracy_per_class(bd_pred, ori_label, num_classes)


'''Defense Effectiveness Rate (DER) ( DER = [max(0,ΔASR) − max(0,ΔACC) + 1]/2, where ΔACC = C-Acc_bd − C-Acc_defnse and ΔASR = ASR_bd − ASR_defnse)'''

def defense_effectiveness_rate(bd_pred, defense_pred, ori_label, target_label):
    '''Compute the defense effectiveness rate'''
    return (max(0, attack_success_rate(bd_pred, target_label) - attack_success_rate(defense_pred, target_label)) + max(0, clean_accuracy(bd_pred, ori_label) - clean_accuracy(defense_pred, ori_label)) + 1) / 2

def defense_effectiveness_rate_per_class(bd_pred, defense_pred, ori_label, target_label, num_classes):
    '''Compute the defense effectiveness rate per class'''
    der = []
    asr_bd = attack_success_rate_per_class(bd_pred, target_label, num_classes)
    asr_defense = attack_success_rate_per_class(defense_pred, target_label, num_classes)
    acc_bd = clean_accuracy_per_class(bd_pred, ori_label, num_classes)
    acc_defense = clean_accuracy_per_class(defense_pred, ori_label, num_classes)
    for i in range(num_classes):
        der.append((max(0, asr_bd[i] - asr_defense[i]) + max(0, acc_bd[i] - acc_defense[i]) + 1) / 2)

'''Robust Improvement Rate (DER) ( DER = [max(0,-ΔRA) − max(0,ΔACC) + 1]/2, where ΔRA = RA_bd − RA_defnse and ΔASR = ASR_bd − ASR_defnse)'''

def robust_improvement_rate(bd_pred, defense_pred, ori_label):
    '''Compute the robust improvement rate'''
    return (max(0, -robust_accuracy(bd_pred, ori_label) + robust_accuracy(defense_pred, ori_label)) + max(0, clean_accuracy(bd_pred, ori_label) - clean_accuracy(defense_pred, ori_label)) + 1) / 2

def robust_improvement_rate_per_class(bd_pred, defense_pred, ori_label, num_classes):
    '''Compute the robust improvement rate per class'''
    rir = []
    ra_bd = robust_accuracy_per_class(bd_pred, ori_label, num_classes)
    ra_defense = robust_accuracy_per_class(defense_pred, ori_label, num_classes)
    acc_bd = clean_accuracy_per_class(bd_pred, ori_label, num_classes)
    acc_defense = clean_accuracy_per_class(defense_pred, ori_label, num_classes)
    for i in range(num_classes):
        rir.append((max(0, -ra_bd[i] + ra_defense[i]) + max(0, acc_bd[i] - acc_defense[i]) + 1) / 2)

def defense_effectiveness_rate_simplied(acc_bd, acc_defnese, asr_bd, asr_defense):
    '''Compute the defense effectiveness rate'''
    return (max(0, asr_bd - asr_defense) - max(0, acc_bd - acc_defnese) + 1) / 2

def robust_improvement_rate_simplied(acc_bd, acc_defnese, ra_bd, ra_defense):
    '''Compute the robust improvement rate'''
    return (max(0, -ra_bd + ra_defense) - max(0, acc_bd - acc_defnese) + 1) / 2