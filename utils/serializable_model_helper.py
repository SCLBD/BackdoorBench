import torch
from torchvision.models import resnet18

def module_without_further_children(net):
    print('notice that for model with parallel module structure, this function cannot the order!!!')
    return [
        module
        for name, module in list(net.named_modules()) if list(module.named_children()).__len__() == 0
    ]

def fix_until_module_name(
        net : torch.nn.Module,
        module_name: str,
) -> list:
    '''

    :param net: model
    :param module_name: module without further children
    :return: list of weights and its require_grad value
    '''
    print('only for model can be serialized!!!(modules have strict order no parallel module structure)')
    for name, module in list(net.named_modules()):
        if list(module.named_children()).__len__() == 0:
            for k,v in module.named_parameters():
                v.requires_grad = False
        if name == module_name:
            break
    return [
        (name , [(k, v.requires_grad) for k,v in module.named_parameters()])
        for name, module in list(net.named_modules()) if list(module.named_children()).__len__() == 0
    ]

def test_():
    from pprint import pprint
    net = resnet18()
    net.train()
    pprint(fix_until_module_name(net, 'layer3.0.conv1'))
    module_without_further_children(net)