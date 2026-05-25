from torch.utils.data import DataLoader
from dataset import AGIQA_1KDataset
from model import IPIQAImageEncoder, IPIQATextEncoder
import torch.nn as nn
import torch
from torchvision import transforms
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.data import random_split
import os
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = "cuda" if torch.cuda.is_available() else "cpu"
dataloader = DataLoader(AGIQA_1KDataset(
   excel_path="D:\IQA_Research\datasets\AGIQA-1K\AIGC_MOS_Zscore.xlsx",
    image_dir='D:/IQA_Research/datasets/AGIQA-1K/file',
    transform=transform),
      batch_size=4, shuffle=True)
image_encoder = IPIQAImageEncoder(device).to(device)
text_encoder = IPIQATextEncoder(device).to(device)
optimizer = torch.optim.Adam(
 image_encoder.parameters(), lr=1e-4
)
for epoch in range(3):
    total_loss = 0.0

    for images, prompts, mos_scores in dataloader:
        images = images.to(device)
        global_feature = image_encoder(images)
        with torch.no_grad():
            text_feature = text_encoder(prompts)
        cos_sim = (global_feature * text_feature).sum(dim=-1)
        loss = 1 - cos_sim.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch + 1}, Avg Pretrain Loss: {avg_loss:.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
    image_encoder.state_dict(),
    f"checkpoints/image2prompt_epoch{epoch+1}.pth"
    )