import random
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register



@register('paired')
class Paired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        img_1, img_2, img_3, img_4 = self.dataset[idx]
      
        if self.inp_size is None:
            crop_1 = img_1
            crop_2 = img_2
            crop_3 = img_3
            crop_4 = img_4

        else:
            w = self.inp_size
           
            x0 = random.randint(40  , img_1.shape[-2] - 20 - w)
            y0 = random.randint(45 , img_1.shape[-1] - 45 - w)
            crop_1 = img_1[:, x0: x0 + w, y0: y0 + w]
            crop_2 = img_2[:, x0: x0 + w, y0: y0 + w]
            crop_3 = img_3[:, x0: x0 + w, y0: y0 + w]
            crop_4 = img_4[:, x0: x0 + w, y0: y0 + w]


        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_1 = augment(crop_1)
            crop_2 = augment(crop_2)
            crop_3 = augment(crop_3)
            crop_4 = augment(crop_4)
  
        return {
            '0': crop_1,
            '1': crop_2,
            '2': crop_3,
            '3': crop_4,
        }

@register('multiparam-paired')
class MultiParamPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        data_1,data_2,data_3,data_4,data_5,data_6,data_7= self.dataset[idx]
        img_1,condition_1 = data_1
        img_2,condition_2 = data_2
        img_3,condition_3 = data_3
        img_4,condition_4 = data_4
        img_5,condition_5 = data_5
        img_6,condition_6 = data_6
        img_7,condition_7 = data_7
  
        if self.inp_size is None:
            crop_1 = img_1
            crop_2 = img_2
            crop_3 = img_3
            crop_4 = img_4
            crop_5 = img_5
            crop_6 = img_6
            crop_7 = img_7
     
        else:
            w = self.inp_size
  
            x0 = random.randint(0 + 20, img_1.shape[-2] - 20 - w)
            y0 = random.randint(0 + 20, img_1.shape[-1] - 20 - w)
            crop_1 = img_1[:, x0: x0 + w, y0: y0 + w]
            crop_2 = img_2[:, x0: x0 + w, y0: y0 + w]
            crop_3 = img_3[:, x0: x0 + w, y0: y0 + w]
            crop_4 = img_4[:, x0: x0 + w, y0: y0 + w]
            crop_5 = img_5[:, x0: x0 + w, y0: y0 + w]
            crop_6 = img_6[:, x0: x0 + w, y0: y0 + w]
            crop_7 = img_7[:, x0: x0 + w, y0: y0 + w]


        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
         
            crop_1 = augment(crop_1)
            crop_2 = augment(crop_2)
            crop_3 = augment(crop_3)
            crop_4 = augment(crop_4)
            crop_5 = augment(crop_5)
            crop_6 = augment(crop_6)
            crop_7 = augment(crop_7)
  
        return {
            '1': [crop_1,condition_1],
            '2': [crop_2,condition_2],
            '3': [crop_3,condition_3],
            '4': [crop_4,condition_4],
            '5': [crop_5,condition_5],
            '6': [crop_6,condition_6],
            '7': [crop_7,condition_7]
        }


