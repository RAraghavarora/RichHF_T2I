from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from models import AestheticScoreModel
from utils import bytes_to_tensor
from scipy import stats
import wandb
import numpy as np

def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * math.sqrt(warmup_steps / step)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = torch.stack([bytes_to_tensor(item) for item in batch['image']]).to(device)
            captions = batch['caption']
            aesthetic_scores = torch.tensor([item for item in batch['aesthetics_score']], dtype=torch.float32).to(device)
            
            predicted_scores = model(images, captions)
            
            all_predictions.extend(predicted_scores.squeeze(1).cpu().numpy())
            all_targets.extend(aesthetic_scores.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    plcc = stats.pearsonr(all_predictions, all_targets)[0]
    srcc = stats.spearmanr(all_predictions, all_targets)[0]
    mse = np.mean((all_predictions - all_targets) ** 2)
    
    return plcc, srcc, mse

def train_model(model, train_loader, val_loader, num_iterations=20000, warmup_steps=2000, base_lr=0.015):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    criterion = nn.MSELoss()
    
    model.train()
    iteration = 0
    best_val_plcc = -1
    eval_every = 10



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
            train_srcc = stats.spearmanr(predicted_scores.detach().cpu().numpy(), aesthetic_scores.cpu().numpy())[0]
            train_plcc = stats.pearsonr(predicted_scores.detach().cpu().numpy(), aesthetic_scores.cpu().numpy())[0]
            if np.nan in train_plcc:
                train_plcc=-1
            else:
                train_plcc = train_plcc.sum()/train_plcc.shape[0]
            wandb.log({
                "train/loss": loss.item(), 
                "learning_rate": lr,
                "train/srcc": train_srcc,
                "train/plcc": train_plcc
            }, step=iteration)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

            if iteration % eval_every == 0:
                val_plcc, val_srcc, val_mse = evaluate_model(model, val_loader, device)
                # print(f"Validation - PLCC: {val_plcc:.4f}, SRCC: {val_srcc:.4f}, MSE: {val_mse:.4f}")
                wandb.log({
                    "val/plcc": val_plcc,
                    "val/srcc": val_srcc,
                    "val/mse": val_mse
                }, step=iteration)
                
                if val_plcc > best_val_plcc:
                    best_val_plcc = val_plcc
                    torch.save(model.state_dict(), 'best_model.pth')
                    # print(f"New best model saved with PLCC: {best_val_plcc:.4f}")
                
                model.train()

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




def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    plcc, srcc, mse = evaluate_model(model, test_loader, device)
    wandb.log({
        "test/plcc": plcc,
        "test/srcc": srcc,
        "test/mse": mse
    })
    
    return plcc, srcc, mse


wandb.init(entity='appliedmachinelearning', project="aesthetic-score-model", config={
        "num_iterations": 20000,
        "warmup_steps": 2000,
        "base_lr": 0.015,
    })
# Initialize model and start training
model = AestheticScoreModel()
rhf_dataset_dict = load_from_disk('./rich_human_feedback_dataset')
train_loader = create_dataloader(rhf_dataset_dict, batch_size=256, split='train')
val_loader = create_dataloader(rhf_dataset_dict, batch_size=256, split='dev')
train_model(model, train_loader, val_loader)

test_loader = create_dataloader(rhf_dataset_dict, batch_size=256, split='test')
plcc, srcc = test_model(model, test_loader)
wandb.finish()