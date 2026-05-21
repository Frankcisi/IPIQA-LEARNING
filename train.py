from torch.utils.data import DataLoader
from dataset import CSIQDataset
from model import CLIPIQA
import torch.nn as nn
import torch
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "D:\\IQA_Research\\datasets\\CSIQ\\dst_imgs_all"
label_path = "D:\\IQA_Research\\datasets\\CSIQ\\csiq_label.txt"
num_epochs=5
loss_list=[]
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
criterion=nn.MSELoss()
dataset = CSIQDataset(image_dir=image_dir, label_path=label_path,transform=transform)
dataloader=DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
model=CLIPIQA(device=device).to(device)
optimizer=torch.optim.Adam(params=model.regressor.parameters(),lr=1e-4)
for epoch in range(num_epochs):
    model.train()
    total_loss=0.0
    for image,score in dataloader:
        image=image.to(device)
        score=score.to(device)
        pred_score=model(image)
        loss=criterion(pred_score,score)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss=total_loss/len(dataloader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    