import os
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from tqdm import tqdm

from PIL import Image
from skimage.color import rgb2lab, rgb2gray, lab2rgb

class ColorizeDataloader(Dataset):
    def __init__(self, path, split, transforms=None):
        self.transforms = transforms
        self.prefix_path = path
        self.split = split
        self.path = pjoin(self.prefix_path, split)
        assert os.path.exists(self.path), f"self.path does not exist {self.path}"
        self.image_fns = os.listdir(self.path)

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, idx):
        assert pjoin(self.path, self.image_fns[idx])
        image = Image.open(pjoin(self.path, self.image_fns[idx])).resize((224, 224)) #Bicubic
        
        gray_image = rgb2gray(image)
        gray_image = np.concatenate(3*[np.expand_dims(gray_image, -1)],-1)
        
        l_image = rgb2lab(gray_image)[:, :, 0]
        l_image = np.expand_dims(l_image/100, -1)
        ab_image = rgb2lab(image)[:, :, 1:]
        ab_image = (ab_image+128)/255

        l_image = l_image.transpose((2, 0, 1))
        ab_image = ab_image.transpose((2, 0, 1))

        l_image = torch.from_numpy(l_image).float()
        ab_image = torch.from_numpy(ab_image).float()

        return l_image, ab_image



if __name__ == "__main__":
    PATH = "/home2/kushagra0301/projects_self/Image_Colorization/dataset"

    train_data = ColorizeDataloader(PATH, "train")
    print("train: ", len(train_data))

    eval_data = ColorizeDataloader(PATH, "eval")
    print("eval: ", len(eval_data))


    for a, b in tqdm(train_data, "train_data"):
        assert a.shape[1:] == b.shape[1:], f"a:{a.shape}, b:{b.shape}"

    for a, b in tqdm(eval_data, "eval_data"):
        assert a.shape[1:] == b.shape[1:], f"a:{a.shape}, b:{b.shape}"


    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=3)
    for a, b in tqdm(train_dataloader, "train_dl"):
        assert a.shape[0] == b.shape[0], f"a:{a.shape}, b:{b.shape}"
    
    eval_dataloader = DataLoader(eval_data, batch_size=128, shuffle=True, num_workers=3)
    for a, b in tqdm(eval_dataloader, "eval_dl"):
        assert a.shape[0] == b.shape[0], f"a:{a.shape}, b:{b.shape}"