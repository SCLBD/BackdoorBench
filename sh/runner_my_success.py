'''
1. All infomation must be logging, DON'T use print !!!!
2. you can have
    pip install nvidia-ml-py3

GPU_allocation_list
# indicate that at each GPU how many jobs. [x,x,x] means KEEP 3 jobs run on x th GPU
'''


GPU_allocation_list = [0, 1, 2]

command_indicator_pair_list_dict = {
    'attack': [
        (
            " python -u ../attack/label_consistent_attack.py --model vgg19 --pratio 0.001 --save_folder_name test_cifar10_label_consistent_vgg19_0_001 --dataset cifar10 --epochs 1",
            "test_cifar10_label_consistent_vgg19_0_001.txt"),
        (
            " python -u ../attack/label_consistent_attack.py --model preactresnet18 --pratio 0.001 --save_folder_name test_cifar10_label_consistent_preactresnet18_0_001 --dataset cifar10 --epochs 1",
            "test_cifar10_label_consistent_preactresnet18_0_001.txt"),
        (
            " python -u ../attack/label_consistent_attack.py --model preactresnet18 --pratio 0.001 --save_folder_name test1_cifar10_label_consistent_preactresnet18_0_001 --dataset cifar10 --epochs 1",
            "test1_cifar10_label_consistent_preactresnet18_0_001.txt"),
        (
            " python -u ../attack/label_consistent_attack.py --model preactresnet18 --pratio 0.001 --save_folder_name test2_cifar10_label_consistent_preactresnet18_0_001 --dataset cifar10 --epochs 1",
            "test2_cifar10_label_consistent_preactresnet18_0_001.txt"),
        (
            " python -u ../attack/label_consistent_attack.py --model preactresnet18 --pratio 0.001 --save_folder_name test3_cifar10_label_consistent_preactresnet18_0_001 --dataset cifar10 --epochs 1",
            "test3_cifar10_label_consistent_preactresnet18_0_001.txt"),
    ],
    'defense': [
        (
            " python -u ../attack/label_consistent_attack.py --model mobilenet_v3_large --pratio 0.005 --save_folder_name test_gtsrb_label_consistent_mobilenet_v3_large_0_005 --dataset gtsrb --epochs 1",
            "test_gtsrb_label_consistent_mobilenet_v3_large_0_005.txt"),
        (
            " python -u ../attack/label_consistent_attack.py --model efficientnet_b3 --pratio 0.005 --save_folder_name test_gtsrb_label_consistent_efficientnet_b3_0_005 --dataset gtsrb --epochs 1",
            "test_gtsrb_label_consistent_efficientnet_b3_0_005.txt"),
    ]

}

import logging, time
import multiprocessing
from multiprocessing import Manager
import os
from pprint import pformat

# set the logger
logFormatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
)
logger = logging.getLogger()
fileHandler = logging.FileHandler("multiprocessing" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)

logging.info(f"GPU_allocation_list:")
logging.info(GPU_allocation_list)
logging.info(f"command_indicator_pair_list_dict:")
logging.info(pformat(command_indicator_pair_list_dict))

def check_finish(file_path):
    """
    check the sign file in the path if the task is finished
    :param file_path:
    :return:
    """
    while not os.path.exists(file_path):
        time.sleep(1)
    return True

with open(f"run_python.sh", 'w') as f:
    f.write(
        f"$1 \n\
echo complete > $2"
    )

def workder(gpu_id_queue, command_without_gpu, indicator_filename):
    # remove the flash in path, to avoid failure in txt generation

    gpu_id = gpu_id_queue.get()

    command_with_gpu = command_without_gpu
    # command_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} " + command_without_gpu
    # command_with_gpu =  command_without_gpu + f" --device cuda:{gpu_id}"

    logging.info(f"gpu {gpu_id} job START, indicator_filename:{indicator_filename}")

    print(command_with_gpu)
#     with open(f'{indicator_filename}_start','w') as f:
#         f.write(f'cd {cd_before_command} \n\
# CUDA_VISIBLE_DEVICES={gpu_id} {command_with_gpu}\n\
# echo complete > {indicator_filename}')
#     os.system(f'{indicator_filename}_start')
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} bash run_python.sh \"{command_with_gpu}\" {indicator_filename}")

    if check_finish(indicator_filename):
        gpu_id_queue.put_nowait(gpu_id)
        logging.info(f"gpu {gpu_id} job END, indicator_filename:{indicator_filename}")
        return

if __name__ == '__main__':

    n_gpu = len(set(GPU_allocation_list))
    n_process = len(GPU_allocation_list)
    logging.info(f"construct pool... n_gpu:{n_gpu}, n_process:{n_process}")
    logging.info(f'GPU_allocation_list:{GPU_allocation_list}')

    for stage_name, job_indicator_pair_list in command_indicator_pair_list_dict.items():

        start_time = time.time()

        logging.info(f"START stage:{stage_name}, job num:{len(job_indicator_pair_list)}")

        pool = multiprocessing.Pool(processes=n_process)

        gpu_id_queue = Manager().Queue()
        for i in GPU_allocation_list:
            gpu_id_queue.put_nowait(str(i))

        for (command_without_gpu, indicator_filename) in job_indicator_pair_list:
            pool.apply_async(
                workder,
                (
                    gpu_id_queue,
                    command_without_gpu,
                    indicator_filename.replace('/', '_'),
                )
            )

        pool.close()
        pool.join()

        end_time = time.time()
        logging.info(f"END stage:{stage_name}, use time:{end_time - start_time}(s)")

    logging.info(f"ALL END, remove all indicator...")

    for stage_name, job_indicator_pair_list in command_indicator_pair_list_dict.items():
        for _, indicator_filename in (job_indicator_pair_list):
            os.system(f"rm -rf {indicator_filename.replace('/', '_')}")