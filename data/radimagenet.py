import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io, transform
import torch
import torch.nn.functional as F


class RadData(Dataset):
    def __init__(self, base_path='/data', split="train", only_task=None):
        self.base_path = base_path
        self.split = split

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}

        normalize = transforms.Normalize(**norm_params)
        
        if split=="train":
            data_split_path = '/rad_train_shuffled.txt'
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            data_split_path = '/rad_test.txt'
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        self.images = []
        self.labels = []
        self.task_labels = []

        self.class_label_to_task_label_dict = {
            0: list(range(0,6)),
            1: list(range(6,34)),
            2: list(range(34,36)),
            3: list(range(36,49)),
            4: list(range(49,67)),
            5: list(range(67,81)),
            6: list(range(81,90)),
            7: list(range(90,115)),
            8: list(range(115,141)),
            9: list(range(141,151)),
            10: list(range(151,165)),
        }
        self.num_task = 11

        if only_task is not None:
            label_list = self.class_label_to_task_label_dict[only_task]
            min_lable = label_list[0]

            with open(data_split_path, 'r') as f: 
                for line in f: 
                    path, label = line.strip().split(' ')
                    if int(label) in label_list:
                        self.images.append(os.path.join(self.base_path, path))
                        self.labels.append(int(label)-min_lable)
                        for each_key in self.class_label_to_task_label_dict.keys():
                            if int(label) in self.class_label_to_task_label_dict[each_key]:
                                self.task_labels.append(torch.tensor(each_key))

        else:
            with open(data_split_path, 'r') as f: 
                for line in f: 
                    path, label = line.strip().split(' ')
                    self.images.append(os.path.join(self.base_path, path))
                    self.labels.append(int(label))
                    for each_key in self.class_label_to_task_label_dict.keys():
                        if int(label) in self.class_label_to_task_label_dict[each_key]:
                            self.task_labels.append(torch.tensor(each_key))
        
        assert len(self.labels) == len(self.task_labels)


    def __getitem__(self, index):
        x = Image.open(self.images[index]).convert('RGB')
        y = self.labels[index]
        task_y = F.one_hot(self.task_labels[index], num_classes=self.num_task) 

        if isinstance(self.transform,list):
            sample1 = self.transform[0](x)
            sample2 = self.transform[1](x)
            return [sample1, sample2], y, task_y

        else:
            x = self.transform(x)
            return x, y, task_y


    def __len__(self):
        return len(self.images)



