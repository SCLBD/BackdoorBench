import os,sys
import numpy as np
import torch


class defense(object):


    def __init__(self,):
        # TODO:yaml config log(测试两个防御方法同时使用会不会冲突)
        print(1)

    def add_arguments(parser):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以实现给参数的功能
        print('You need to rewrite this method for passing parameters')
    
    def set_result(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以读取攻击的结果
        print('You need to rewrite this method to load the attack result')
        
    def set_trainer(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现整合训练模块的功能
        print('If you want to use standard trainer module, please rewrite this method')
    
    def set_logger(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现存储log的功能
        print('If you want to use standard logger, please rewrite this method')

    def denoising(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def mitigation(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def inhibition(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def defense(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def detect(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

