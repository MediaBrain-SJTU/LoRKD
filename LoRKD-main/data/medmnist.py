import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io, transform
import torch
import torch.nn.functional as F
import copy


class MedMnistData(Dataset):
    def __init__(self, base_path='/DATASET', split="train", only_task=None, task_start=None, task_end=None):
        self.base_path = base_path

        self.info_list = ['pathmnist', 'octmnist', 'organcmnist', 'bloodmnist', 'breastmnist', \
                'tissuemnist', 'dermamnist', 'organamnist', 'organsmnist', 'pneumoniamnist']
        self.label_num = [9, 4, 11, 8, 2, 8, 7, 11, 11, 2] 

        if only_task is not None:
            self.info_list = self.info_list[only_task:only_task+1]
            self.label_num = self.label_num[only_task:only_task+1]
        if task_start is not None:
            self.info_list = self.info_list[task_start:task_end+1]
            self.label_num = self.label_num[task_start:task_end+1]

        self.split = split

        self.npz_file_list = []
        self.class_label_to_task_label_dict = {}
        task_id = 0
        max_label = 0
        for each_info_th in range(len(self.info_list)):
            each_info = self.info_list[each_info_th]
            label_num =  self.label_num[each_info_th]
            self.class_label_to_task_label_dict[task_id] = []

            npz_file = np.load(os.path.join(self.base_path, "{}.npz".format(each_info)))
            self.npz_file_list.append(npz_file)

            if self.split == 'train':
                imgs = copy.deepcopy(npz_file['train_images'])
                labels = copy.deepcopy(npz_file['train_labels'])
            elif self.split == 'val':
                imgs = copy.deepcopy(npz_file['val_images'])
                labels = copy.deepcopy(npz_file['val_labels'])
            elif self.split == 'test':
                imgs = copy.deepcopy(npz_file['test_images'])
                labels = copy.deepcopy(npz_file['test_labels'])
            else:
                raise ValueError

            if each_info_th == 0:
                self.images = imgs
                self.labels = labels
                self.task_labels = np.array([task_id]*len(labels))
            else:
                if len(imgs.shape) == 3:
                    imgs = imgs[:,:,:,np.newaxis]
                    imgs = np.tile(imgs, (1,1,1,3))
                self.images = np.concatenate((self.images, imgs), axis=0)

                self.labels = np.concatenate((self.labels, (labels + max_label)), axis=0)

                self.task_labels = np.concatenate((self.task_labels, np.array([task_id]*len(labels))), axis=0)

            label_set = np.unique(labels)
            assert len(label_set) == label_num
            self.class_label_to_task_label_dict[task_id] = label_set + max_label
            max_label = max_label + len(label_set)
            task_id = task_id + 1


        self.num_task = len(self.info_list)

        if self.split == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])  


        self.task_labels = torch.tensor(self.task_labels)

    def __getitem__(self, index):
        x = Image.fromarray(self.images[index]).convert('RGB')
        y = self.labels[index].astype(int)
        task_y = F.one_hot(self.task_labels[index], num_classes=self.num_task) 

        x = self.transform(x)
        return x, y, task_y


    def __len__(self):
        return len(self.images)



