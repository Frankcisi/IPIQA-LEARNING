import torch
import torch.nn as nn
class IQAModel(nn.Module):
   def __init__(self):
      super().__init__()
      self.image_encoder=nn.Sequential(
        nn.Flatten(),
        nn.Linear(224*224*3, 128),
        nn.ReLU()
      )
      self.text_encoder=nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU()
      )
      self.regressor=nn.Linear(128+16, 1)
   def forward(self,image,prompts):
      imgae_feat=self.image_encoder(image)
      prompts_lengths=[len(p) for p in prompts]
      prompts_lengths=torch.tensor(prompts_lengths, dtype=torch.float32,device=image.device).view(-1, 1)
      text_feat=self.text_encoder(prompts_lengths)
      fusion_featrue=torch.cat([imgae_feat,text_feat],dim=1)
      score=self.regressor(fusion_featrue)
      return score