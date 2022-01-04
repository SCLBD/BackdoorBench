
import torchvision

import torch.utils.data as data

class CelebA_attr(data.Dataset):
    def __init__(self, data_root, split):
        self.dataset = torchvision.datasets.CelebA(root=data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        # self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        # input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)