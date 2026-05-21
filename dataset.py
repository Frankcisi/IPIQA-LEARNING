import pandas as pd
import os
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class CSIQDataset(Dataset):
    def __init__(self,image_dir,label_path,transform):
        self.image_dir=image_dir
        self.label_path=label_path
        self.transform=transform
        self.samples=[]
        with open(label_path,'r',encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line=="":
                    continue
                image_name,score=line.split()
                self.samples.append((image_name,float(score)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        image_name,score=self.samples[idx]
        image_path=os.path.join(self.image_dir,image_name)
        image=Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        return image,torch.tensor(score,dtype=torch.float32)
if __name__ == "__main__":
    image_dir = "D:\\IQA_Research\\datasets\\CSIQ\\dst_imgs_all"
    label_path = "D:\\IQA_Research\\datasets\\CSIQ\\csiq_label.txt"
    dataset = CSIQDataset(image_dir=image_dir, label_path=label_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,nums_workers=0)
    for images, scores in dataloader:
        print("Images shape:", images.shape)
        print("Scores shape:", scores.shape)
        print("Scores:", scores)
        break