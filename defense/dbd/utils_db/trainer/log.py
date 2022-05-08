import os

import pandas as pd
import torch
from tabulate import tabulate


def tabulate_step_meter(batch_idx, num_batches, num_intervals, meter_list, logger):
    """Tabulate current average value of meters every `step_interval`.

    Args:
        batch_idx (int): The batch index in an epoch.
        num_batches (int): The number of batch in an epoch.
        num_intervals (int): The number of interval to tabulate.
        meter_list (list or tuple of AverageMeter): A list of meters.
        logger (logging.logger): The logger.
    """
    step_interval = int(num_batches / num_intervals)
    if batch_idx % step_interval == 0:
        step_meter = {"Iteration": ["{}/{}".format(batch_idx, num_batches)]}
        for m in meter_list:
            step_meter[m.name] = [m.batch_avg]
        table = tabulate(step_meter, headers="keys", tablefmt="github", floatfmt=".5f")
        if batch_idx == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        logger.info(table)


def tabulate_epoch_meter(elapsed_time, meter_list, logger):
    """Tabulate total average value of meters every epoch.

    Args:
        eplased_time (float): The elapsed time of a epoch.
        meter_list (list or tuple of AverageMeter): A list of meters.
        logger (logging.logger): The logger.
    """
    epoch_meter = {m.name: [m.total_avg] for m in meter_list}
    epoch_meter["time"] = [elapsed_time]
    table = tabulate(epoch_meter, headers="keys", tablefmt="github", floatfmt=".5f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
    logger.info(table)


def result2csv(result, log_dir):
    for k in result.keys():
        file_path = os.path.join(log_dir, k + ".csv")
        if not os.path.exists(file_path):
            df = pd.DataFrame.from_records([result[k]])
            df.to_csv(file_path, index=False)
        else:
            with open(file_path) as f:
                df = pd.read_csv(f)
                df = df.append(result[k], ignore_index=True)
                df.to_csv(file_path, index=False)


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=None):
        self.name = name
        self.reset()

    def reset(self):
        self.batch_avg = 0
        self.total_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, avg, n=1):
        self.batch_avg = avg
        self.sum += avg * n
        self.count += n
        self.total_avg = self.sum / self.count


class Record(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = torch.zeros(self.size)

    def update(self, batch_data):
        self.data[self.ptr : self.ptr + len(batch_data)] = batch_data
        self.ptr += len(batch_data)
