import sys, logging
sys.path.append('../')
import torch
import time
from functools import wraps, partial

# from api.composite_log import composite_log


class resource_check(object):
    """ count how much time used in this process,
     and during this process how torch.cuda.max_memory_allocated() is changing at beginning and end

     can be applied to both normal function and instance method"""
    def __init__(self, func):
        wraps(func)(self)
        self.func = func

    def __call__(self, *args, **kwargs):

        start_time = time.time()
        if torch.cuda.is_available():
            start_gpu_usage = torch.cuda.max_memory_allocated()/(10**9)

        return_value_from_func = self.func(*args, **kwargs)

        end_time = time.time()
        print(f'process:{self.func.__name__}')
        print(f'    time use :{end_time - start_time}(s)')
        if torch.cuda.is_available():
            end_gpu_usage = torch.cuda.max_memory_allocated()/(10 ** 9)
            max_gpu_ram = torch.cuda.get_device_properties(0).total_memory/(10 ** 9)
            logging.info(f'    GPU:{start_gpu_usage:8.6e}(G) -> {end_gpu_usage:8.6e}(G), change:{(end_gpu_usage - start_gpu_usage):8.6e}, total GPU RAM: {max_gpu_ram:8.6e}')

        return return_value_from_func

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)

