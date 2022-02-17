import numpy as np
import torch
import torch.utils.data
from utils_meta import load_model_setting, epoch_meta_train_oc, epoch_meta_eval_oc
from meta_classifier import MetaClassifierOC
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack to evaluate. M: modification attack; B: blending attack.')
parser.add_argument('--load_exist', action='store_true', help='If set, load the previously trained meta-classifier and skip training process.')

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.troj_type in ('M', 'B'), 'unknown trojan pattern'

    GPU = True
    N_REPEAT = 5
    N_EPOCH = 10

    TRAIN_NUM = 2048
    VAL_NUM = 256
    TEST_NUM = 256

    save_path = './meta_classifier_ckpt/%s_oc.model'%args.task
    shadow_path = './shadow_model_ckpt/%s/models'%args.task

    Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task)
    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.cuda()
            inp_std = inp_std.cuda()
    print ("Task: %s; target Trojan type: %s; input size: %s; class num: %s"%(args.task, args.troj_type, input_size, class_num))

    train_dataset = []
    for i in range(TRAIN_NUM):
        x = shadow_path + '/shadow_benign_%d.model'%i
        train_dataset.append((x,1))

    test_dataset = []
    for i in range(TEST_NUM):
        x = shadow_path + '/target_benign_%d.model'%i
        test_dataset.append((x,1))
        x = shadow_path + '/target_troj%s_%d.model'%(args.troj_type, i)
        test_dataset.append((x,0))

    AUCs = []
    for i in range(N_REPEAT):
        shadow_model = Model(gpu=GPU)
        target_model = Model(gpu=GPU)
        meta_model = MetaClassifierOC(input_size, class_num, gpu=GPU)
        if inp_mean is not None:
            #Initialize the input using data mean and std
            init_inp = torch.zeros_like(meta_model.inp).normal_()*inp_std + inp_mean
            meta_model.inp.data = init_inp
        else:
            meta_model.inp.data = meta_model.inp.data

        if not args.load_exist:
            print ("Training One-class Meta Classifier %d/%d"%(i+1, N_REPEAT))
            optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
            for _ in tqdm(range(N_EPOCH)):
                epoch_meta_train_oc(meta_model, shadow_model, optimizer, train_dataset, is_discrete=is_discrete)
                torch.save(meta_model.state_dict(), save_path+'_%d'%i)
        else:
            print ("Evaluating One-class Meta Classifier %d/%d"%(i+1, N_REPEAT))
            meta_model.load_state_dict(torch.load(save_path+'_%d'%i))
        test_info = epoch_meta_eval_oc(meta_model, shadow_model, test_dataset, is_discrete=is_discrete, threshold='half')
        print ("\tTest AUC:", test_info[1])
        AUCs.append(test_info[1])
        
    AUC_mean = sum(AUCs) / len(AUCs)
    print ("Average detection AUC on %d meta classifier: %.4f"%(N_REPEAT, AUC_mean))
