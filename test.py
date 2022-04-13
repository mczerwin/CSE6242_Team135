from PIL import Image
import os
import numpy as np
import urllib.request
import pickle
from torchvision import transforms as T
import torchvision.models as models
import torch

from dataSet import CustomDataset


def load_image(image_file):
    img = np.array(Image.open(image_file))
    return img


def load_model(filename):
    return torch.load(open(filename, 'rb'), map_location=torch.device('cpu'))


#Input csv and img as CustomDataset
infer = CustomDataset(csv_path='./row.csv', image_folder='./images', column_label='Pawpularity')
print(infer)
#Load data via torch
img = torch.utils.data.DataLoader(infer)
model = load_model('./eff_net.sav')
for input, label in img:
    score = model(input)


print('After inference')
print(score)
print(score[0][0])




