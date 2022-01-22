import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from .data import Data
from .trigger import Trigger
from .triggerpool import TriggerPool

def display_batch(batch):
    # for imagenet
    def normalize(tensor, mean, std):
        return tensor.mul(torch.tensor(std).view(1,3,1,1)).add(torch.tensor(mean).view(1,3,1,1))
    batch1 = normalize((batch), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    fig = plt.figure(figsize=(20,20))
    plt.imshow(np.vstack(np.hsplit(np.hstack(batch1.numpy().transpose(0,2,3,1)),8)))
    plt.show()
    return

def prepare_data(name, type_, batchsize=None):
    if name == 'cifar':
        # sa ddsas
        data_path = './pytorch-cifar/data'
        transforms = torchvision.transforms

        if type_ == 'train':
            if batchsize == None: batchsize = 128
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

        else: # testset
            if batchsize == None: batchsize = 100
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
            
            
    elif name == 'cifar100':
        data_path = './pytorchcifar100/data'
        transforms = torchvision.transforms
        if type_ == 'train':
            if batchsize == None: batchsize = 128
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

        else: # testset
            if batchsize == None: batchsize = 100
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
    else: # imagenet
        transforms = torchvision.transforms
        
        if type_ == 'train':
            data_path = './imagenet/train'
            dataset = torchvision.datasets.ImageFolder(data_path, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)
            
        else: # 'val'
            data_path = './imagenet/val'
            dataset = torchvision.datasets.ImageFolder(data_path, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)
    
    return Data(name, dataset, dataloader)

def find_trigger(model, data):
    pass

def find_trigger_distribution(model, data, num_triggers, threshold):
    """find some triggers for a model"""
    def generate_random_trigger():
        pattern = np.ones((3,3,3)) * 0.5 # 3*3 gray square
        return Trigger(model.name, pattern, target=0, type_=0)
        
    pool = TriggerPool()
    pool.add(find_trigger(model, data))
    while len(pool.success_triggers(threshold)) < num_triggers:
        pool.test(model, data)
        pool.expand(5)
    print("Found %d triggers in %d, threshold %.1f." % (
        len(pool.success_triggers(threshold)), len(pool.triggers), float(threshold)))
    return pool.success_triggers(threshold)

def remove_backdoor(model, data, trigger_test, triggers=None, num_epochs=1, mode="eval"):
    """retrain the model to remove backdoor"""
    def apply_random_trigger(inputs, triggers, ratio=1):
        """pick one trigger and apply to the whole batch"""
        trigger = triggers[int(np.random.random() * len(triggers))]
        return trigger.apply_batch(inputs, ratio)
    
    def get_flat_activation(model, inputs):
        """get model activation and flatten it into 2D: [Batchsize, Number of activations]"""
        model.net.forward(inputs)
        return torch.cat([a[1].view(a[1].size(0), -1) for a in model.activation.items()], 1)
        
    def get_activation_loss(model, inputs, model1, backdoor_inputs):
        """get MSE loss between clean and backdoor activations"""
        loss = torch.nn.MSELoss()
        clean_act = get_flat_activation(model, inputs).detach() # prevent unnessary grad computation
        backdoor_act = get_flat_activation(model1, backdoor_inputs)
        return loss(backdoor_act, clean_act)    
        
    def report_triggers_accuracy(model, data, triggers):
        acc = []
        for trigger in triggers:
            acc.append(model.test(data, 0.1, trigger).accuracy())
        print(acc)
    
    def apply_activation_hook(model): ### to be moved to Class Model
        """add forward hooks to obtain conv layers' activations"""
        def filter_layers(model, type_):
            """list all layers of a certain type"""
            def filter_children(obj, fun):
                l = list(obj.children())
                return (([obj] if fun(obj) else []) if len(l) == 0
                        else [j for i in l for j in filter_children(i, fun)])
            return filter_children(model.net, lambda x: isinstance(x, type_))
        
        model.activation = {}
        def hook_wrapper(model, name):
            """store outputs to dict"""
            def hook(self, input, output):
                model.activation[name] = output
            return hook
        
        conv_layers = filter_layers(model, torch.nn.Conv2d)
        for i, layer in enumerate(conv_layers):
            layer.register_forward_hook(hook_wrapper(model, "conv"+str(i)))
        return model
        
    if triggers == None:
        triggers = find_trigger_distribution(model, data, num_triggers=20, threshold=99)
    model1 = model.duplicate()
    optimizer = torch.optim.SGD(model1.net.parameters(), lr=1e-3, momentum=0.9)
    apply_activation_hook(model)
    apply_activation_hook(model1)
    for epoch in range(num_epochs):
        if mode == "eval":
            model.net.eval()
            model1.net.eval() 
        else:
            model.net.train()
            model1.net.train()# forward 1 epoch in train mode can remove the backdoor!!!
        for batch_idx, (inputs, targets) in enumerate(data.dataloader):
            backdoor_inputs = apply_random_trigger(inputs.clone().detach(), triggers, ratio=0.5)
            inputs, backdoor_inputs = inputs.to(model.device), backdoor_inputs.to(model1.device)
            loss = get_activation_loss(model, inputs, model1, backdoor_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch %d:" % (epoch))
        model1.test(data, 0.5).report_accuracy()
        model1.test(data, 0.5, trigger_test).report_accuracy()
        report_triggers_accuracy(model1, data, triggers)
    return model1