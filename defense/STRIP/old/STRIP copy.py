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

import torch
import os
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms

import math

from defenses.STRIP.config import get_argument

import sys

sys.path.insert(0, "../..")
from classifier_models import PreActResNet18, ResNet18
from defenses.STRIP.dataloader import get_dataloader, get_dataset
from utils.utils import progress_bar
from networks.models import NetC_MNIST, Normalizer, Denormalizer


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
            add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

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

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)

#########从这开始是我写的部分
def strip_v1(opt,arg,model,train_data,train_data_clean,test_data_clean,test_data_bd,trainset):
    #输入模型是已经load的模型
    netC = model
    netC.requires_grad_(False)
    netC.eval()
    netC.to(opt.device)

    testset = get_dataset(opt, train=False)
    opt.bs = opt.n_test
    test_dataloader_backdoor = test_data_clean
    denormalizer = Denormalizer(opt)

    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []


    if opt.n_test < arg.batch_size:
        bd_inputs, targets = next(iter(test_dataloader_backdoor))
        bd_inputs = bd_inputs.to(opt.device)
        bd_inputs = denormalizer(bd_inputs) * 255.0
        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))

        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)
            progress_bar(index, opt.n_test)
    else:
        step = math.ceil(opt.n_test/arg.batch_size)
        for i in range(step-1):
            bd_inputs, targets = next(iter(test_dataloader_backdoor))
            bd_inputs = bd_inputs.to(opt.device)
            bd_inputs = denormalizer(bd_inputs) * 255.0
            bd_inputs = bd_inputs.detach().cpu().numpy()
            bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))

            for index in range(arg.batch_size):
                background = bd_inputs[index]
                entropy = strip_detector(background, testset, netC)
                list_entropy_trojan.append(entropy)
                progress_bar(index, opt.n_test)
        bd_inputs, targets = next(iter(test_dataloader_backdoor))
        bd_inputs = bd_inputs.to(opt.device)
        bd_inputs = denormalizer(bd_inputs) * 255.0
        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))

        for index in range(opt.n_test-(step-1)*arg.batch_size):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)
            progress_bar(index, opt.n_test)

    # Testing with clean data
    for index in range(opt.n_test):
        background, _ = testset[index]
        entropy = strip_detector(background, testset, netC)
        list_entropy_benign.append(entropy)

    return list_entropy_trojan, list_entropy_benign

def strip_defense(arg,model,train_data,train_data_clean,test_data_clean,test_data_bd,trainset):
    opt = get_argument().parse_args()
    ####参数平移
    opt.data_root = arg.data_root
    opt.input_width = arg.input_width
    opt.input_height = arg.input_height
    opt.input_channel = arg.input_channel
    opt.num_classes = arg.num_classes
    opt.dataset = arg.dataset
    opt.device = arg.device

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(opt.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip_v1(opt,arg,model,train_data,train_data_clean,test_data_clean,test_data_bd,trainset)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        bd_model = True
    else:
        bd_model = False

    result = {}
    result.update({'lists_entropy_trojan': lists_entropy_trojan})
    result.update({'lists_entropy_benign': lists_entropy_benign})
    result.update({'backdoor_model' : bd_model})
    return result



if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/AC/config/config.yaml", 'r') as stream: 
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
        result_defense = ac(args,result,config)
    else:
        print("There is no target model")