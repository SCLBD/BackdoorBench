'''
@inproceedings{gao2019strip,
  title={Strip: A defence against trojan attacks on deep neural networks},
  author={Gao, Yansong and Xu, Change and Wang, Derui and Chen, Shiping and Ranasinghe, Damith C and Nepal, Surya},
  booktitle={Proceedings of the 35th Annual Computer Security Applications Conference},
  pages={113--125},
  year={2019}
}

code : https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses/STRIP
'''

import argparse
import torch
import os
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import math

#from defenses.STRIP.config import get_argument

import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
import yaml
from utils.aggregate_block.dataset_and_transform_generate import get_transform

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC

sys.path.insert(0, "../..")

class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class STRIP:
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0])
            add_image = self.tran(Image.fromarray(add_image.astype(np.uint8)))
            #add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, opt, tran = None):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.device = opt.device
        self.tran = tran

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x


    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)

#########从这开始是我写的部分
def strip_v1(opt,result,config):
    #输入模型是已经load的模型
    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    netC = model
    netC.requires_grad_(False)
    netC.eval()

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy())[0:opt.n_test])
    y = result['clean_test']['y'][0:opt.n_test]
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    testset = data_set_o
    
    opt.bs = opt.n_test
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy())[0:opt.n_test])
    y = result['bd_train']['y'][0:opt.n_test]
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    test_dataloader_backdoor = data_set_o

    # STRIP detector
    strip_detector = STRIP(opt,tran)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []


    
    for i in range(opt.n_test):  # type: ignore
        bd_inputs, targets = test_dataloader_backdoor[i]
        background = bd_inputs
        entropy = strip_detector(background, testset, netC)
        list_entropy_trojan.append(entropy)

    # Testing with clean data
    for index in range(opt.n_test):
        background, _ = testset[index]
        entropy = strip_detector(background, testset, netC)
        list_entropy_benign.append(entropy)

    return list_entropy_trojan, list_entropy_benign

def strip_defense(arg,result,config):

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(arg.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip_v1(arg,result,config)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, arg.detection_boundary))
    if min_entropy < arg.detection_boundary:
        bd_model = True
    else:
        bd_model = False

    result = {}
    result.update({'lists_entropy_trojan': lists_entropy_trojan})
    result.update({'lists_entropy_benign': lists_entropy_benign})
    result.update({'backdoor_model' : bd_model})
    return result

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    ####strip
    parser.add_argument("--n_sample", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--detection_boundary", type=float)  # According to the original paper
    parser.add_argument("--test_rounds", type=int)

    parser.add_argument("--s", type=float)
    parser.add_argument("--k", type=int)  # low-res grid size
    parser.add_argument(
        "--grid-rescale", type=float
    )  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98

    arg = parser.parse_args()

    print(arg)
    return arg

if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/STRIP/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    args.checkpoint_save = os.getcwd() + '/record/defence/ac/' + args.dataset + '.tar'
    args.log = 'saved/log/log_' + args.dataset + '.txt'

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    args.save_path = save_path
    result = torch.load(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = strip_defense(args,result,config)
    else:
        print("There is no target model")