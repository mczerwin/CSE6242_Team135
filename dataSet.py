import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

image_height, image_width = 64, 64
class CustomDataset(Dataset):
    def __init__(self, images):
        # images: array of PIL image objects
        # self.df = pd.read_csv(csv_path)
        # self.image_folder = Path(image_folder)
        self.images = images
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_height, image_width)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # assert column_label in self.df.columns
        # self.column_label = column_label
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # filename = self.df.loc[index]['Id'] + '.jpg'
        # image_path = self.image_folder.joinpath(filename)
        image = self.transform(self.images[index])
        
        # label = self.df.loc[index][self.column_label]
        
        # return image, label
        return image