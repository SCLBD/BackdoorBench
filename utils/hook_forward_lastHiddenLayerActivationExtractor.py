# @misc{uozbulak_pytorch_vis_2021,
#   author = {Utku Ozbulak},
#   title = {PyTorch CNN Visualizations},
#   year = {2019},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/utkuozbulak/pytorch-cnn-visualizations}},
#   commit = {53561b601c895f7d7d5bcf5fbc935a87ff08979a}
# }

import torch
from typing import Tuple

class lastHiddenActivationExtractor():
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.lastLinearInput = None

        self.model.eval()
        self.hook_layers()

    def hook_layers(self) -> None:
        def hook_function(module, input, output):
            self.lastLinearInput = input #  THIS IS A TUPLE !!!!

        # Register hook to the first layer
        # first_layer = list(self.model.features._modules.items())[0][1]
        lastLinearLayer = [i for i in list(self.model.modules()) if isinstance(i, torch.nn.Linear)][-1]
        lastLinearLayer.register_forward_hook(hook_function)

    def generateLastLinearInput(self, input_tensor_images, device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(device)
        output_before_softmax = self.model(input_tensor_images)
        return self.lastLinearInput[0], output_before_softmax

if __name__ == '__main__':
    from torchvision.models import resnet18
    a = resnet18(pretrained=False, num_classes = 10)
    from dataset_preprocess.cifar10_preprocess import CIFAR10, data_transforms_cifar10
    train_trans, test_trans = data_transforms_cifar10()
    b = CIFAR10('../data/cifar10/cifar-10-batches-py' ,train = False, transform= test_trans)
    from torch.utils.data import DataLoader
    bdl = DataLoader(b, batch_size= 2, shuffle= False)
    c = lastHiddenActivationExtractor(a)
    print(c.generateLastLinearInput(iter(bdl).__next__()[0], device = torch.device('cpu')))
    print(1)