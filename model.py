import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
class IPIQAImageEncoder(nn.Module):
    def __init__(self, device):
        super(IPIQAImageEncoder, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = True
    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        global_feature = F.normalize(image_features, dim=-1)
        return global_feature
class IPIQATextEncoder(nn.Module):
    def __init__(self, device):
        super(IPIQATextEncoder, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, prompts):
        text_tokens = clip.tokenize(prompts).to(next(self.parameters()).device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features