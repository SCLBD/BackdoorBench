import sys
import os

sys.path.append(os.getcwd())

from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from analysis.visual_utils import *
import yaml
import torch
import torch.nn as nn
import matplotlib as mlp
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

def set_devices(args):
    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )
    return device

def main():
    # Basic setting: args
    args = get_args()
    args.yaml_path = args.yaml_path
    print("Load config from {}".format(args.yaml_path))
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args = preprocess_args(args)
    fix_random(int(args.random_seed))

    args.device = set_devices(args)
    save_path_attack = "./record/" + args.result_file_attack  
    visual_save_path = save_path_attack + f"/defense/{args.result_file_defense}"
    # Load result
    result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

    selected_classes = np.arange(args.num_classes)

    args.visual_dataset = 'clean_train'
    args.n_sub = args.ratio*len(result_attack["clean_train"]) 

    # Create dataset
    if args.visual_dataset == 'mixed':
        bd_test_with_trans = result_attack["bd_test"]
        visual_dataset = generate_mix_dataset(
            bd_test_with_trans, args.target_class, args.pratio, selected_classes, max_num_samples=args.n_sub)
    elif args.visual_dataset == 'clean_train':
        clean_train_with_trans = result_attack["clean_train"]
        visual_dataset = generate_clean_dataset(
            clean_train_with_trans, selected_classes, max_num_samples=args.n_sub)
    elif args.visual_dataset == 'clean_test':
        clean_test_with_trans = result_attack["clean_test"]
        visual_dataset = generate_clean_dataset(
            clean_test_with_trans, selected_classes, max_num_samples=args.n_sub)
    elif args.visual_dataset == 'bd_train':
        bd_train_with_trans = result_attack["bd_train"]
        visual_dataset = generate_bd_dataset(
            bd_train_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub)
    elif args.visual_dataset == 'bd_test':
        bd_test_with_trans = result_attack["bd_test"]
        visual_dataset = generate_bd_dataset(
            bd_test_with_trans, args.target_class, selected_classes, max_num_samples=args.n_sub)
    else:
        assert False, "Illegal vis_class"

    poi_list = np.array(get_poison_indicator_from_bd_dataset(visual_dataset))
    print(f'Create visualization dataset with \n \t Dataset: {args.visual_dataset} \n \t Number of samples: {len(visual_dataset)}  \n \t Selected classes: {selected_classes} \n \t Number of poison samples: {poi_list.sum()}')

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        visual_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # Create denormalization function
    for trans_t in data_loader.dataset.wrap_img_transform.transforms:
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            
    # Load model
    model_visual = generate_cls_model(args.model, args.num_classes)
    model_visual.load_state_dict(result_attack["model"])
    print(f"Load model {args.model} from {args.result_file_attack}")
    model_visual.to(args.device)

    # !!! Important to set eval mode !!!
    model_visual.eval()

    defense_dict = torch.load(visual_save_path + f"/plug_layer.pt")
    module_name = defense_dict["target_layer"]
    print(module_name)
    plug_layer_dict = defense_dict["plug_model"]
    args.model_name = defense_dict["plug_name"]

    
    def output_feature_hook(module, input_, output_): ## sum pooling over spatial dimensions -> bs x c
        activation = None
        global out_feature_vector
        global input_feature_vector
        # access the layer output and convert it to a feature vector
        input_feature_vector = input_[0]
        out_feature_vector = output_
        if activation is not None:
            out_feature_vector = activation(out_feature_vector)
        if out_feature_vector.dim() > 2:
            out_feature_vector = torch.sum(torch.flatten(out_feature_vector, 2), 2) 
        else:
            out_feature_vector = out_feature_vector
        return None

    def input_feature_hook(module, input_):

        if args.model_name in ['twoconv','cnn', 'onlyconv','convbn','lightconv']:
            modified_input = plug_layer(input_[0])
            return modified_input
        elif args.model_name in ['linear', 'linear_light']:
            modified_input = input_[0] * w1 + b1
        elif args.model_name == 'mlp':
            modified_input = relu(input_[0] * w1 + b1) * w2 + b2
        return modified_input

    def evaluation(model_visual):
        # Create dataset
        visual_dataset = result_attack["clean_test"]
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            visual_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
        target_class = args.target_class
        criterion = torch.nn.CrossEntropyLoss()
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        target_correct, target_total = 0, 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model_visual(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            target_correct += torch.sum((torch.argmax(outputs[:], dim=1) == target_class)*(labels[:] == target_class))
            target_total += torch.sum(labels[:] == target_class)

            total_clean_test += inputs.shape[0]
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
        print('Test Acc: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
        print('Test Acc (Target only): {:.3f}%({}/{})'.format(target_correct/target_total*100.0, target_correct, target_total))
        clean_acc = avg_acc_clean
        target_acc = (target_correct/target_total*100.0).item()
        # Create dataset
        visual_dataset = result_attack["bd_test"]

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            visual_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
        # Load model
        target_class = args.target_class 
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        target_correct, target_total = 0, 0
        total_ra = 0
        for i, (inputs, labels, *others) in enumerate(data_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            true_y = others[-1].to(args.device)
            outputs = model_visual(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_ra += torch.sum(torch.argmax(outputs[:], dim=1) == true_y[:])
            total_clean_test += inputs.shape[0]
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
        print('Test ASR: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
        avg_ra = float(total_ra.item() * 100.0 / total_clean_test)
        print('Test RA: {:.3f}%({}/{})'.format(avg_ra, total_ra, total_clean_test))
        test_asr = avg_acc_clean
        return clean_acc, target_acc, test_asr, avg_ra

    ### Collect info of the target layer
    module_dict = dict(model_visual.named_modules())
    target_layer = module_dict[module_name]

    random_batch, *other_info = next(iter(data_loader))
    random_batch = random_batch.to(args.device)
    # Collect random feature vector
    h_out = target_layer.register_forward_hook(output_feature_hook)
    logits = model_visual(random_batch)
    base_feature_clean = out_feature_vector.detach().cpu().numpy()
    base_clean_input = input_feature_vector.detach().cpu().numpy()

    h_out.remove()
    size = base_clean_input.shape
    # change the first element of tuple to 1
    size = list(size)
    use_bias = False
    chan = size[1]
    use_residule = False
    kernel_size = 1

    if args.model_name == 'mlp':
        weight_size = [1,size[1],size[2],size[3]]
        w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
        b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
        relu = nn.ReLU()
        w2 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
        b2 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
        plug_layer = [w1,b1,w2,b2]
    elif args.model_name == 'onlyconv':
        plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias))
        plug_layer[0].weight.data.fill_(0)
        for i in range(chan):
            plug_layer[0].weight.data[i, i, 0, 0] = 1
    elif args.model_name in ['lightconv','convbn']:
        plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias), nn.BatchNorm2d(chan))
        plug_layer[0].weight.data.fill_(0)
        for i in range(chan):
            plug_layer[0].weight.data[i, i, 0, 0] = 1
    elif args.model_name == 'linear':
        weight_size = [1,size[1],size[2],size[3]]
        w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
        b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
        plug_layer = [w1,b1]
    elif args.model_name == 'linear_light':
        weight_size = [1,size[1],1,1]
        w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
        b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
        plug_layer = [w1,b1]
    elif args.model_name == 'twoconv':
        plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias), nn.BatchNorm2d(chan),  nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias),nn.BatchNorm2d(chan))
    elif args.model_name == 'cnn':
        plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias), nn.BatchNorm2d(chan),  nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias))

    plug_layer.load_state_dict(plug_layer_dict)
    h_in = target_layer.register_forward_pre_hook(input_feature_hook)
    
    plug_layer.eval()
    model_visual.eval()
    model_visual.to(args.device)
    plug_layer.to(args.device)
    clean_acc, target_acc, test_asr, avg_ra = evaluation(model_visual)
    print(f"Result: {clean_acc}|{target_acc}|{test_asr}|{avg_ra}")
    h_in.remove()
   
    

if __name__ == '__main__':
    main()

## run:
## python utils/defense_utils/npd/evaluation.py   --result_file_attack xx --result_file_defense xx

