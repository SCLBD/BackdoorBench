'''
1. All infomation must be logging, DON'T use print !!!!
2. you can have
    pip install nvidia-ml-py3

GPU_allocation_list
# indicate that at each GPU how many jobs. [x,x,x] means KEEP 3 jobs run on x th GPU
'''
import logging
import multiprocessing
import os
import shutil
import socket
import time
from multiprocessing import Manager
from pprint import pformat

# scp 需要的包 pip install paramiko pip install scp
import paramiko
from scp import SCPClient

def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip  

    

GPU_allocation_list = [0, 2]

datasets=['gtsrb']
pratios=['1']
backbones=["preactresnet18"]
attacks=["badnet"]
defenses=['ft'] 

#传数据相关
is_transfer=True
remote_path='/workspace/bdzoo_record'
local_ip=get_host_ip()

#gtsrb,tiny
remote_ip2='10.26.1.68'
username2='zhuzihao'
password2='zzh961011'

#cifar10,cifar100
remote_ip1='10.26.1.69'
username1='weishaokui'
password1='weishaokui'

if 'cifar10' in datasets or 'cifar100' in datasets:
    remote_ip=remote_ip1
    username=username1
    password=password1
elif 'tiny' in datasets or 'gtsrb' in datasets:
    remote_ip=remote_ip2
    username=username2
    password=password2

def myscp(local_path,remote_path):

    # 建立 SSH 连接
    ssh = paramiko.SSHClient()
    # 允许连接不在know_hosts文件中的主机
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=remote_ip, port=22, username=username, password=password)

    # SCPClient 使用 paramiko 传输作为参数
    scp = SCPClient(ssh.get_transport())
    scp.put(local_path, remote_path, recursive=True)
    scp.close()
    ssh.close()

def mycp(local_path,remote_path):
    if os.path.exists(remote_path):
        shutil.rmtree(remote_path)
    shutil.copytree(local_path, remote_path) 

def transfer_all_defense_folders(datasets,pratios,backbones,attacks):
    for dataset in datasets:
        for pratio in pratios:
            for backbone in backbones:
                for attack in attacks:
                    for defense in defenses:
                        attack_folder=f'{dataset}_{backbone}_{attack}_0_{pratio}'
                        defense_floder=f'{attack_folder}/{defense}'
                        # 如果本地有defense_result，则传到数据中心
                        if os.path.exists(f'record/{defense_floder}/defense_result.pt'):
                            local_path=f'record/{defense_floder}'
                            # 如果ip一样，则cp到数据中心
                            if local_ip==remote_ip:
                                remote_path=f'/workspace/bdzoo_record/{attack_folder}/{defense}'
                                print(f"cp from {local_path} ===> {remote_path}")
                                mycp(local_path,remote_path)
                            # 如果ip不一样，则scp到数据中心
                            else:
                                remote_path=f'/workspace/bdzoo_record/{attack_folder}/'
                                print(f"scp from {local_ip} : {local_path} ===> {remote_ip} : {remote_path}")
                                myscp(local_path,remote_path)

                        


def generate_attack_commands(datasets,pratios,backbones,attacks):
    commands=[]
    for dataset in datasets:
        for pratio in pratios:
            for backbone in backbones:
                for attack in attacks:

                    result_folder=f'{dataset}_{backbone}_{attack}_0_{pratio}'
                  
                    
                    if not os.path.exists(f'record/{result_folder}/attack_result.pt'): # 如果record 下attack/attack_result.pt 目录 不存在，则才运行attack命令

                        if os.path.exists(f'record/{result_folder}'):
                            shutil.rmtree(f'record/{result_folder}')

                        if attack in ['badnet','blended','sig','ssba','lc','lf']:
                            command=f'python -u ./attack/{attack}_attack.py --yaml_path ../config/attack/{attack}/{dataset}.yaml --model {backbone} --pratio  0.{pratio} --save_folder_name {result_folder}  --dataset {dataset}'
                        
                        if attack == 'inputaware':
                            command=f'python -u ./attack/{attack}_attack.py --yaml_path ../config/attack/{attack}/{dataset}.yaml --model {backbone} --pratio 0.{pratio} --save_folder_name  {result_folder}  --dataset  {dataset}  --checkpoints ../record/{result_folder}/checkpoints/ --temps ../record/{result_folder}/temps --random_seed 10'
                        
                        if attack == 'wanet':
                            command=f'python -u ./attack/{attack}_attack.py --yaml_path ../config/attack/{attack}/{dataset}.yaml --model {backbone} --pratio 0.{pratio} --save_folder_name {result_folder}  --dataset {dataset} --checkpoints ../record/{result_folder}/checkpoints/ --temps ../record/{result_folder}/temps'
                        
                        commands.append((command,f'{result_folder}.txt'.replace('/','_')))
                        
    return commands

def generate_defense_commands(datasets,pratios,backbones,defenses,attacks):
    commands=[]
    for dataset in datasets:
        for pratio in pratios:
            for backbone in backbones:
                for attack in attacks:
                    for defense in defenses:

                        result_folder=f'{dataset}_{backbone}_{attack}_0_{pratio}'
                        defense_result=f'{result_folder}/{defense}/defense_result.pt'

                        if not os.path.exists(f'record/{defense_result}'): # 如果defense_result.pt 不存在，则才运行defense命令

                            if defense in ['ac' ,'abl','spectral','dbd']:
                                command=f"python -u defense/{defense}/{defense}.py --model {backbone} --result_file {result_folder}  --dataset {dataset} --yaml_path config/defense/{defense}/{dataset}.yaml"
                            
                            if defense in ['ft','anp','nad','nc','fp']:
                                command=f"python -u defense/{defense}/{defense}.py --model {backbone} --result_file {result_folder}  --dataset {dataset}  --yaml_path config/defense/{defense}/config.yaml --index config/defense/index/{dataset}_index.txt"
                            
                            commands.append((command,f'{defense_result}.txt'.replace('/','_')))
    return commands



command_indicator_pair_list_dict = {
    'attack': generate_attack_commands(datasets,pratios,backbones,attacks),
    'defense': generate_defense_commands(datasets,pratios,backbones,defenses,attacks)
}



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
    
    if is_transfer==True:
        print('>>>>>>>>正在传送数据<<<<<<<<<')
        transfer_all_defense_folders(datasets,pratios,backbones,attacks)

    
