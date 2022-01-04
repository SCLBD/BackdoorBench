
import numpy as np
from PIL import Image

import torch.utils.data as data

class Tiny(data.Dataset):
    def __init__(self, data_root, train):
        super(Tiny, self).__init__()
        self.data_root = data_root
        self.id_dict = self._get_id_dictionary()
        if train:
            self.images, self.labels = self._get_data_train_list()
            # print(f'image shape: {(self.images).size}; lables shape: {self.labels.shape}')
        else:
            self.images, self.labels = self._get_data_test_list()
            # print(f'image shape: {(self.images).shape}; lables shape: {self.labels.shape}')
        # self.transforms = transforms

    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(self.data_root + '/wnids.txt', 'r')):
            line = line.split()[0]
            id_dict[line] = i
        return id_dict

    def _get_data_train_list(self):
        # print('starting loading data')
        train_data, train_labels = [], []
        for key, value in self.id_dict.items():
            train_data += [self.data_root + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in
                           range(500)]
            train_labels += [value] * 500
        return np.array(train_data), np.array(train_labels)

    def _get_data_test_list(self):
        test_data, test_labels = [], []
        for line in open(self.data_root + '/val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append((self.data_root + '/val/images/{}'.format(img_name)))
            test_labels.append(self.id_dict[class_id])
        return np.array(test_data), np.array(test_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # image = self.transforms(image)
        label = self.labels[index]
        return image, label