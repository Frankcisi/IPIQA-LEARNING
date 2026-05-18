import pandas as pd
import os
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
class IQADataset(Dataset):
    def __init__(self):
        self.data=pd.read_csv('data/labels.csv')
        image_dir="data/images"
        self.trnsform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image_name=self.data.iloc[index,0]
        image_path=os.path.join("data/images",image_name)
        image=Image.open(image_path).convert('RGB')
        prompt=self.data.iloc[index,1]
        score=self.data.iloc[index,2]
        image=self.trnsform(image)
        score=torch.tensor(score)
        return image, prompt, score