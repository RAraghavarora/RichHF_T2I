from datasets import load_from_disk, load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from models import AestheticScoreModel
from utils import bytes_to_tensor, tokenize_captions, get_lr
from scipy import stats
import wandb
import numpy as np
from transformers import AutoTokenizer


def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)
            aesthetic_scores = batch['aesthetics_score'].to(device)
            
            predicted_scores = model(images, captions).squeeze(1)
            
            all_predictions.extend(predicted_scores.cpu().numpy())
            all_targets.extend(aesthetic_scores.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    plcc = stats.pearsonr(all_predictions, all_targets)[0]
    srcc = stats.spearmanr(all_predictions, all_targets)[0]
    mse = np.mean((all_predictions - all_targets) ** 2)
    
    return plcc, srcc, mse

def train_model(model, train_loader, val_loader, num_iterations=20000, warmup_steps=1000, base_lr=0.001):
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

            # images = torch.stack([bytes_to_tensor(item) for item in batch['image']]).to(device)
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)
            # images = torch.stack([bytes_to_tensor(item['image']) for item in batch]).to(device)
            aesthetic_scores = batch['aesthetics_score'].to(device)
            
            # Update learning rate
            lr = get_lr(iteration + 1, warmup_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            predicted_scores = model(images, captions).squeeze(1)
            loss = criterion(predicted_scores, aesthetic_scores)
            loss.backward()
            optimizer.step()
            train_srcc = stats.spearmanr(predicted_scores.detach().cpu().numpy(), aesthetic_scores.cpu().numpy())[0]
            train_plcc = stats.pearsonr(predicted_scores.detach().cpu().numpy(), aesthetic_scores.cpu().numpy())[0]

            if np.isnan(train_plcc) or np.isnan(train_srcc):
                print("NaN discovered")
                import pdb; pdb.set_trace()
                print("NaN discovered")

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


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = bytes_to_tensor(item['image'])
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        caption = tokenizer(item['caption'], return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        return {
            'image': image,
            'caption': caption['input_ids'].squeeze(),
            # 'attention_mask': caption['attention_mask'].squeeze(),
            # 'artifact_score': torch.tensor(item['artifact_score']),
            # 'misalignment_map': torch.tensor(item['misalignment_map']),
            # 'misalignment_score': torch.tensor(item['misalignment_score']),
            # 'overall_score': torch.tensor(item['overall_score']),
            'aesthetics_score': torch.tensor(item['aesthetics_score']),
            # 'artifact_map': torch.tensor(item['artifact_map']),
            # 'prompt_misalignment_label': torch.tensor(item['prompt_misalignment_label']),
            # 'filename': item['filename']
        }


def create_dataloader(dataset, split, batch_size=32, shuffle=True):
    custom_dataset = CustomDataset(dataset[split])
    return DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
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
try:
    rhf_dataset_dict = load_from_disk('./rich_human_feedback_dataset')
except FileNotFoundError:
    rhf_dataset_dict = load_dataset('RAraghavarora/RichHumanFeedback')

train_loader = create_dataloader(rhf_dataset_dict, batch_size=32, split='train')
val_loader = create_dataloader(rhf_dataset_dict, batch_size=8, split='dev')
test_loader = create_dataloader(rhf_dataset_dict, batch_size=32, split='test')
print("loaded")
train_model(model, train_loader, val_loader)

plcc, srcc = test_model(model, test_loader)
wandb.finish()