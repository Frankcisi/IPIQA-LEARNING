from torch.utils.data import DataLoader
from dataset import CSIQDataset
from model import CLIPIQA
import torch.nn as nn
import torch
from torchvision import transforms
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.data import random_split
best_srcc=0
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
train_size=int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])
train_loader=DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader=DataLoader(test_dataset,batch_size=4, shuffle=False, num_workers=0)
model=CLIPIQA(device=device).to(device)
optimizer=torch.optim.Adam(params=model.regressor.parameters(),lr=1e-4)
for epoch in range(num_epochs):
    model.train()
    total_loss=0.0
    for image,score in train_loader:
        image=image.to(device)
        score=score.to(device)
        pred_score=model(image)
        loss=criterion(pred_score,score)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss=total_loss/len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # ===== evaluate =====
    model.eval()
    pred_scores=[]
    mos_scores=[]
    with torch.no_grad():
        for image,score in test_loader:
            image=image.to(device)
            score=score.to(device)
            pred_score=model(image)
            pred_score = pred_score.detach().cpu().view(-1).tolist()
            score = score.detach().cpu().view(-1).tolist()
            pred_scores.extend(pred_score)
            mos_scores.extend(score)

        srcc=spearmanr(pred_scores,mos_scores)[0]
        plcc=pearsonr(pred_scores,mos_scores)[0]
        krcc=kendalltau(pred_scores,mos_scores)[0]
        print(f"SRCC: {srcc:.4f}")
        print(f"PLCC: {plcc:.4f}")
        print(f"KRCC: {krcc:.4f}")
        if srcc > best_srcc:
            best_srcc = srcc
            torch.save(
                model.state_dict(),
                "checkpoints/exp001_frozen_clip_regression_best_model.pth"
            )

            print("Best model saved!")