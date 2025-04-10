import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(transforms.ToTensor()(
                    Image.open(file)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_0, root_path_1,root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolder(root_path_0, **kwargs)
        self.dataset_2 = ImageFolder(root_path_1, **kwargs)
        self.dataset_3 = ImageFolder(root_path_2, **kwargs)
        self.dataset_4 = ImageFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx]

@register('multiparam-image-folder')
class MultiParamImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        self.condition = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(transforms.ToTensor()(
                    Image.open(file)))
            self.cond = torch.tensor(self.get_condition(root_path)).float()
            self.condition.append(self.cond)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        condition = self.condition[idx % len(self.files)]
        return x, condition
    
    def get_condition(self, path):
        if path[-1] == '1':
            return [24.79, 9.62, 10.55]
        elif path[-1] == '2':
            return [24.79, 10.42, 0]
        elif path[-1] == '3':
            return [67.77, 10.42, 0]
        elif path[-1] == '4':
            return [67.77, 62.52, 0]
        elif path[-1] == '5':
            return [80, 10.42, 0]
        elif path[-1] == '6':
            return [80, 39.92, 0]
        elif path[-1] == '7':
            return [81.42, 104.2, 0]
        elif path[-1] == '8':
            return [80, 99.98, 10.3]
        elif path[-1] == '9':
            return [80, 99.98, 20.25]
        
@register('multiparam-paired-image-folders')
class MultiParamPairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2,root_path_3, root_path_4, root_path_5, root_path_6, root_path_7,**kwargs):
        self.dataset_1 = MultiParamImageFolder(root_path_1, **kwargs)
        self.dataset_2 = MultiParamImageFolder(root_path_2, **kwargs)
        self.dataset_3 = MultiParamImageFolder(root_path_3, **kwargs)
        self.dataset_4 = MultiParamImageFolder(root_path_4, **kwargs)
        self.dataset_5 = MultiParamImageFolder(root_path_5, **kwargs)
        self.dataset_6 = MultiParamImageFolder(root_path_6, **kwargs)
        self.dataset_7 = MultiParamImageFolder(root_path_7, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx], self.dataset_5[idx], self.dataset_6[idx], self.dataset_7[idx]