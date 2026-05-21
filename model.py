import torch
import torch.nn as nn
import clip
class CLIPIQA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.regressor=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, image):
        with torch.no_grad():
            image_features=self.clip_model.encode_image(image)
            image_features=image_features.float()
        scores=self.regressor(image_features)
        return scores.squeeze(1)