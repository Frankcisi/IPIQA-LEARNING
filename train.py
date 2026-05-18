from torch.utils.data import DataLoader
from dataset import IQADataset  
from model import IQAModel
import torch.nn as nn
import torch
epochs = 5
model = IQAModel()
dataset = IQADataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    total_loss = 0

    for images, prompts, scores in loader:
        pred = model(images, prompts)
        scores = scores.float().view(-1, 1)
        print(pred.shape, scores.shape)
        loss = criterion(pred, scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Average Loss: {avg_loss:.4f}")