import torch, logging

def partial_load(net : torch.nn.Module, given_state_dict):
    net_state_dict = net.state_dict()
    for k, v in net_state_dict.items():
        if k in given_state_dict and net_state_dict[k].shape == v.shape:
            net_state_dict[k] = v
        else:
            logging.warning(f"{k} doesn't find matched key or shape is different.\n original_shape:{net_state_dict[k].shape}, load_shape:{v.shape}")

    net.load_state_dict(net_state_dict)
    return net


def test_1():
    from torchvision.models.resnet import resnet18
    a = resnet18(num_classes = 44)
    b = resnet18(num_classes = 10)
    partial_load(a, b.state_dict())
    print(list(a.state_dict().items())[-1][1].shape)