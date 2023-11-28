from PIL import Image
import os
from diffusion_sde.configs.config import CFGS
from torch.utils.data import Dataset
# from imageNamesBUSI import train_img_list as train_img_list
# from imageNamesBUSI import train_mask_list as train_mask_list
# from imageNamesBUSI import train_label_list as train_label_list

from imageNamesBrain import train_img_list as brain_train_img_list
from imageNamesBrain import train_mask_list as brain_train_mask_list

import numpy as np
import cv2

# class datasets(Dataset):
#     def __init__(self, root_img):
#         self.root_img = root_img
#         self.transform = None
#         self.images = os.listdir(root_img)
#         self.length_dataset = len(self.images)
        
#     def __len__(self):
#         return self.length_dataset

#     def __getitem__(self, index):
#         img = self.images[index % self.length_dataset]
    
#         img_path = os.path.join(self.root_img, img)
#         img = Image.open(img_path).convert("RGB")
        
#         if self.transform:
#             img = self.transform(img)

#         return img      
    
# class datasets(Dataset):
#     def __init__(self, resize=128) -> None:
#         r = len(train_img_list)
#         real_ratio = 1
#         self.img_dir = train_img_list
#         self.mask_dir = train_mask_list
#         # print(len(train_mask_list), len(supple_mask_list))

        
#         self.img_list = []
#         self.mask_list = []

#         n = len(self.img_dir)
#         pos_num, neg_num = 0, 0
#         count = 0
#         # onehot_encoder = OneHotEncoder(sparse=False)
#         for i in range(n):
            
#             # print(self.img_dir[i], self.mask_dir[i])
#             img = cv2.imread(self.img_dir[i])
#             seg = cv2.imread(self.mask_dir[i])
            
#             img = cv2.resize(img, (resize, resize))
#             seg = cv2.resize(seg, (resize, resize), interpolation=0)

#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             seg = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)

#             seg = seg / 255.
#             img = img / 255.

#             self.img_list.append(img)
#             self.mask_list.append(seg)


#         print(len(self.img_list))
#         print(len(self.mask_list))


#     def __len__(self):
#         return len(self.img_list)
    
#     def __getitem__(self, index):
#         # print(index)
#         img = self.img_list[index]
#         mask = self.mask_list[index]

#         img, mask = np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)
#         # mask = np.expand_dims(mask, axis=0)
#         img, mask = img.astype(np.float32), mask.astype(np.float32)

#         pair_data = np.concatenate([img, mask], axis=0)
#         # img = np.concatenate([img, img, img], axis=0)
#         return pair_data



class Braindatasets(Dataset):
    def __init__(self, resize=128) -> None:
        self.img_dir = brain_train_img_list
        self.mask_dir = brain_train_mask_list

        print(len(self.img_dir), len(self.mask_dir))
        n = len(self.img_dir)
        
        self.img_list = []
        self.mask_list = []
        
        for i in range(n):

            img = cv2.imread(self.img_dir[i])
            seg = cv2.imread(self.mask_dir[i])

            img = cv2.resize(img, (resize, resize))
            seg = cv2.resize(seg, (resize, resize), interpolation=0)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            seg = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)
            
            seg = seg / 255.
            img = img / 255.
            
            self.img_list.append(img)
            self.mask_list.append(seg)
            # print(np.unique(img), np.unique(seg))
        

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        # print(index)
        img = self.img_list[index]
        mask = self.mask_list[index]

        img, mask = np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)
        # mask = np.expand_dims(mask, axis=0)
        img, mask = img.astype(np.float32), mask.astype(np.float32)

        pair_data = np.concatenate([img, mask], axis=0)
        # img = np.concatenate([img, img, img], axis=0)
        return pair_data
    


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.subset)