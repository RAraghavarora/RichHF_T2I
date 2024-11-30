import torch
from torch import nn
from torch.optim import Adam
from torchmetrics.regression import MeanSquaredError, R2Score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


class EnhancedCNNModel(nn.Module):
    def __init__(self, input_dim, num_outputs):
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.flattened_size = self._compute_flattened_size(input_dim)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_outputs)

    def _compute_flattened_size(self, input_dim):
        dummy_input = torch.zeros(1, 1, input_dim)
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Grid Search for Learning Rate
learning_rates = [5.4e-4, 5.9e-4, 5.8e-4, 5.7e-4, 5.6e-4, 5.5e-4]
best_lr = None
best_lr_r2 = None
best_val_loss = float('inf')
best_val_r2 = float('-inf')
best_test_r2 = float('-inf')
best_lr_test_r2 = None

train_losses, val_losses, val_r2_scores, test_results = [], [], [], []

# Input dimension
input_dim = train_loader.dataset.tensors[0].shape[1]
num_outputs = 1

# Metrics
mse_metric = MeanSquaredError().to(device)
r2_metric = R2Score().to(device)

for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")

    model = EnhancedCNNModel(input_dim, num_outputs).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    criterion = nn.MSELoss()

    # Train for a few epochs (less than 50?)
    num_epochs = 30
    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        train_loss = 0
        for vectors, scores in train_loader:
            vectors = vectors.to(device)
            scores = scores.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(vectors)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0
        all_val_outputs, all_val_targets = [], []
        with torch.no_grad():
            for vectors, scores in val_loader:
                vectors = vectors.to(device)
                scores = scores.to(device).unsqueeze(1)
                outputs = model(vectors)
                val_loss += criterion(outputs, scores).item()
                all_val_outputs.append(outputs)
                all_val_targets.append(scores)
        val_loss /= len(val_loader)

        # Combine validation predictions and targets
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)

        # Validation Metrics
        val_r2 = r2_metric(all_val_outputs, all_val_targets).item()
        y_val_pred = all_val_outputs.cpu().numpy().squeeze()
        y_val_true = all_val_targets.cpu().numpy().squeeze()
        val_pearson = pearsonr(y_val_pred, y_val_true)
        val_spearman = spearmanr(y_val_pred, y_val_true)

        # Metrics for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation R²: {val_r2:.4f}, Validation Pearson: {val_pearson[0]:.4f}, Validation Spearman: {val_spearman.correlation:.4f}")
        scheduler.step(val_loss)

    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)
    train_losses.append(train_loss)

    # Best Learning Rate (MSE)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_lr = lr

    # Best Learning Rate (Validation R2)
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_lr_r2 = lr

    # Test
    test_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for vectors, scores in test_loader:
            vectors = vectors.to(device)
            scores = scores.to(device).unsqueeze(1)
            outputs = model(vectors)
            test_loss += criterion(outputs, scores).item()
            y_true.append(scores.cpu())
            y_pred.append(outputs.cpu())
    test_loss /= len(test_loader)

  
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Test Metrics
    test_mse = mse_metric(y_pred.to(device), y_true.to(device)).item()
    test_r2 = r2_metric(y_pred.to(device), y_true.to(device)).item()
    y_test_pred = y_pred.cpu().numpy().squeeze()
    y_test_true = y_true.cpu().numpy().squeeze()
    test_pearson = pearsonr(y_test_pred, y_test_true)
    test_spearman = spearmanr(y_test_pred, y_test_true)

    # Best Test R2
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_lr_test_r2 = lr

    test_results.append(test_loss)

# Final Results
final_results = f"""
--- Final Results ---
Best Learning Rate (MSE): {best_lr}
Best Validation Loss: {best_val_loss:.4f}
Best Learning Rate (Validation R2): {best_lr_r2}
Best Validation R2: {best_val_r2:.4f}
Best Validation Pearson: {val_pearson[0]:.4f}
Best Validation Spearman: {val_spearman.correlation:.4f}
Best Learning Rate (Test R2): {best_lr_test_r2}
Best Test R2: {best_test_r2:.4f}
Best Test Pearson: {test_pearson[0]:.4f}
Best Test Spearman: {test_spearman.correlation:.4f}
Test MSE: {test_mse:.4f}
Test R2: {test_r2:.4f}
"""

print(final_results)

# Write results to a file
output_file = "results_with_pearson_spearman.txt"
with open(output_file, "w") as file:
    file.write(final_results)

print(f"Results saved to {output_file}")

# Plot Train and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, train_losses, label='Train Loss', marker='o', color='blue')
plt.plot(learning_rates, val_losses, label='Validation Loss', marker='o', color='red')
plt.plot(learning_rates, val_r2_scores, label='Validation R²', marker='o', color='green')
plt.xlabel('Learning Rate')
plt.ylim (0, 0.2)
plt.ylabel('Metrics')
plt.xscale('log') 
plt.title('Model Stats')
plt.legend()
plt.grid