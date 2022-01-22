import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
import os
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack type. M: modification attack; B: blending attack.')
if __name__ == '__main__':
    args = parser.parse_args()
    assert args.troj_type in ('M', 'B'), 'unknown trojan pattern'

    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
    TARGET_NUM = 256
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting(args.task)
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the attacker:",target_indices)

    SAVE_PREFIX = './shadow_model_ckpt/%s'%args.task
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX+'/models'):
        os.mkdir(SAVE_PREFIX+'/models')

    all_target_acc = []
    all_target_acc_mal = []

    for i in range(TARGET_NUM):
        model = Model(gpu=GPU)
        atk_setting = random_troj_setting(args.troj_type)
        trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=target_indices, need_pad=need_pad)
        trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)
        testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
        testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)

        train_model(model, trainloader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP), is_binary=is_binary, verbose=False)
        save_path = SAVE_PREFIX+'/models/target_troj%s_%d.model'%(args.troj_type, i)
        torch.save(model.state_dict(), save_path)
        acc = eval_model(model, testloader_benign, is_binary=is_binary)
        acc_mal = eval_model(model, testloader_mal, is_binary=is_binary)
        print ("Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s"%(acc, acc_mal, save_path, datetime.now()))
        p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
        print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))
        all_target_acc.append(acc)
        all_target_acc_mal.append(acc_mal)

    log = {'target_num':TARGET_NUM,
           'target_acc':sum(all_target_acc)/len(all_target_acc),
           'target_acc_mal':sum(all_target_acc_mal)/len(all_target_acc_mal)}
    log_path = SAVE_PREFIX+'/troj%s.log'%args.troj_type
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)
