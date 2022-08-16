# idea: select model you use in training and the trainer (the warper for training process)

import sys, logging
sys.path.append('../../')

import torch 
import torchvision.models as models
from torchvision.models.resnet import resnet18, resnet34
from typing import Optional
from torchvision.transforms import Resize

from utils.trainer_cls import ModelTrainerCLS

try:
    from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b3
except:
    logging.warning("efficientnet_b0,b3 fails to import, plz update your torch and torchvision")
try:
    from torchvision.models import mobilenet_v3_large
except:
    logging.warning("mobilenet_v3_large fails to import, plz update your torch and torchvision")

try:
    from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
except:
    logging.warning("vit fails to import, plz update your torch and torchvision")


#trainer is cls
def generate_cls_model(
    model_name: str,
    num_classes: int = 10,
    image_size : int = 32,
    **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''

    logging.info("image_size ONLY apply for vit!!!\nIf you use vit make sure you set the image size!")

    if model_name == 'resnet18':
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'preactresnet18':
        assert len(kwargs.keys()) == 0  # means we do NOT allow any kwargs in this case !!!
        logging.warning('Make sure you want PreActResNet18, which is NOT resnet18.')
        from models.preact_resnet import PreActResNet18
        net = PreActResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        net = resnet34(num_classes=num_classes, **kwargs)
    elif model_name == 'alexnet':
        net = models.alexnet(num_classes= num_classes, **kwargs)
    elif model_name == "vgg11":
        net = models.vgg11(num_classes= num_classes, **kwargs)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes= num_classes, **kwargs)
    elif model_name == 'vgg19':
        net = models.vgg19(num_classes = num_classes, **kwargs)
    elif model_name == 'squeezenet1_0':
        net = models.squeezenet1_0(num_classes= num_classes, **kwargs)
    elif model_name == 'densenet161':
        net = models.densenet161(num_classes= num_classes, **kwargs)
    elif model_name == 'inception_v3':
        net = models.inception_v3(num_classes= num_classes, **kwargs)
    elif model_name == 'googlenet':
        net = models.googlenet(num_classes= num_classes, **kwargs)
    elif model_name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v2':
        net = models.mobilenet_v2(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(num_classes= num_classes, **kwargs)
    elif model_name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes= num_classes, **kwargs)
    elif model_name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(num_classes= num_classes, **kwargs)
    elif model_name == 'mnasnet1_0':
        net = models.mnasnet1_0(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b0':
        net = efficientnet_b0(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = efficientnet_b3(num_classes= num_classes, **kwargs)
    elif model_name.startswith("vit"):
        logging.info("All vit model use the default pretrain and resize to match the input shape!")
        if model_name == 'vit_b_16':
            net = vit_b_16(
                pretrained = True,
                  **kwargs
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_b_32':
            net = vit_b_32(
                pretrained = True,
                  **kwargs
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_l_16':
            net = vit_l_16(
                pretrained = True,
                  **kwargs
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_l_32':
            net = vit_l_32(
                pretrained = True,
                  **kwargs
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def generate_cls_trainer(
        model,
        attack_name : Optional[str] = None,
        amp : bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer