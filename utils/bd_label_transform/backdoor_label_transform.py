# This script include all-to-one and all-to-all attack

import sys, logging
sys.path.append('../')
import random

class AllToOne_attack(object):
    '''
    idea : any label -> fix_target
    '''
    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--target_label (only one)', type=int,
                            help='target label')
        return parser
    def __init__(self, target_label):
        self.target_label = target_label
    def __call__(self, original_label, original_index = None, img = None):
        return self.poison_label(original_label)
    def poison_label(self, original_label):
        return self.target_label

class AllToAll_shiftLabelAttack(object):
    '''
    idea : any label -> (label + fix_shift_amount) % num_classses
    '''
    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--shift_amount', type=int,
                            help='shift_amount of all_to_all attack')
        parser.add_argument('--num_classses', type=int,
                            help='total number of labels')
        return parser
    def __init__(self, shift_amount, num_classses):
        self.shift_amount = shift_amount
        self.num_classses = num_classses
    def __call__(self, original_label, original_index = None, img = None):
        return self.poison_label(original_label)
    def poison_label(self, original_label):
        label_after_shift = (original_label + self.shift_amount)% self.num_classses
        return label_after_shift
