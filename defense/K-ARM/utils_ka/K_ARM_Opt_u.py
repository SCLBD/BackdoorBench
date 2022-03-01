# K-Arm Optimization
########################################################################################################################################
### K_Arm_Opt functions load data based on different trigger types, then create an instance of K-Arm scanner and run optimization    ###
### It returns the target-victim pair and corresponding pattern, mask and l1 norm of the mask                                        ###
########################################################################################################################################




import torch 
from torchvision import transforms
#from dataset import CustomDataSet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils_ka.K_Arm_Scanner_u import *





def K_Arm_Opt(args,result,target_classes_all,triggered_classes_all,trigger_type,model,direction):

    device = args.device
    

    data_loader_arr = []
    if triggered_classes_all is None:
        transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
        x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy()))
        y = result['bd_train']['y']
        data_set = torch.utils.data.TensorDataset(x,y)
        dataset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_set,
            poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=transform,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        data_loader_arr.append(data_loader)
    
    else:
        for i in range(len(target_classes_all)):
            transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
            x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy()))
            y = result['bd_train']['y']
            data_set = torch.utils.data.TensorDataset(x,y)
            dataset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_set,
                poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=transform,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            dataset.subset([j for j,v in enumerate(y) if v==triggered_classes_all[i]])
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
            data_loader_arr.append(data_loader)


    k_arm_scanner = K_Arm_Scanner(model,args)


    if args.single_color_opt == True and trigger_type == 'polygon_specific':
        pattern = torch.rand(1,args.input_channel,1,1).to(device)
    
    else:
        pattern = torch.rand(1,args.input_channel,args.input_width,args.input_height).to(device)
        #only for r1
        #pattern = torch.rand(1,args.channels,1,1).to(device)
    pattern = torch.clamp(pattern,min=0,max=1)



    #K-arms Bandits
    if trigger_type == 'polygon_global':
        #args.early_stop_patience = 5

        mask = torch.rand(1,args.input_width,args.input_height).to(device)
        

    elif trigger_type == 'polygon_specific':

        if args.central_init:
            mask = torch.rand(1,args.input_width,args.input_height).to(device) * 0.001
            mask[:,112-25:112+25,112-25:112+25] = 0.99
        
        else:
            mask = torch.rand(1,args.input_width,args.input_height).to(device)

    
    mask = torch.clamp(mask,min=0,max=1)

    if args.num_classes == 1:
        start_label_index = 0
    else:
        #start_label_index = torch.randint(0,args.num_classes-1,(1,))[0].item()
        start_label_index = 0

    pattern, mask, l1_norm, total_times = k_arm_scanner.scanning(target_classes_all,data_loader_arr,start_label_index,pattern,mask,trigger_type,direction)
    index = torch.argmin(torch.Tensor(l1_norm))


    '''
    print(pattern[index]) 
    print(mask[index])
    print(total_times[index])
    '''
    

    if triggered_classes_all is None:
        target_class =  target_classes_all[index]
        triggered_class = 'all'

    else:
        target_class = target_classes_all[index]
        triggered_class = triggered_classes_all[index]



    return l1_norm[index], mask[index],target_class,triggered_class,total_times[index]