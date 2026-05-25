import pandas as pd
import os
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_excel(
    "D:\IQA_Research\datasets\AGIQA-1K\AIGC_MOS_Zscore.xlsx"
    
)
class AGIQA_1KDataset(Dataset):
    def __init__(self, excel_path, image_dir, transform=None):
        self.df = pd.read_excel(excel_path)
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["Image"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        prompt = self.df.iloc[idx]["Prompt"]
        mos_score = self.df.iloc[idx]["MOS"]
        if self.transform:
            image = self.transform(image)
        return image, prompt, mos_score

