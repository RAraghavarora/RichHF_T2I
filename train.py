from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from models import AestheticScoreModel
from utils import bytes_to_tensor

def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * math.sqrt(warmup_steps / step)

def train_model(model, train_loader, num_iterations=20000, warmup_steps=2000, base_lr=0.015):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    criterion = nn.MSELoss()
    
    model.train()
    iteration = 0
    while iteration < num_iterations:
        for batch in train_loader:
            if iteration >= num_iterations:
                break

            images = torch.stack([bytes_to_tensor(item) for item in batch['image']]).to(device)
            captions = batch['caption']
            # images = torch.stack([bytes_to_tensor(item['image']) for item in batch]).to(device)
            aesthetic_scores = torch.tensor([item for item in batch['aesthetics_score']], dtype=torch.float32).to(device)
            
            # Update learning rate
            lr = get_lr(iteration + 1, warmup_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            predicted_scores = model(images, captions)
            loss = criterion(predicted_scores.squeeze(1), aesthetic_scores)
            loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
            
            iteration += 1

# Usage example:
def create_dataloader(dataset, batch_size=256, split='train'):
    return DataLoader(
        dataset[split],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

# Initialize model and start training
model = AestheticScoreModel()
rhf_dataset_dict = load_from_disk('./rich_human_feedback_dataset')
train_loader = create_dataloader(rhf_dataset_dict, batch_size=256, split='train')
train_model(model, train_loader)