import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.swa_utils import AveragedModel

class ModelNN(nn.Module):

    def __init__(self, input_size):
        super(ModelNN, self).__init__()
        self._hidden1 = nn.Linear(input_size, 128)
        self._ln1     = nn.LayerNorm(128)
        self._hidden2 = nn.Linear(128, 64)
        self._ln2     = nn.LayerNorm(64)
        self._output  = nn.Linear(64, 1)
        self._relu    = nn.ReLU()
        self._dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self._dropout(self._relu(self._ln1(self._hidden1(x))))
        x = self._dropout(self._relu(self._ln2(self._hidden2(x))))
        return self._output(x)

class Trainer:

    def __init__(self, model, device, learning_rate, pos_weight, swa_start=3, swa_end=None):
        self._device = device
        self._model = model.to(device)
        self._swa_start = swa_start
        self._swa_end = swa_end
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self._optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
        )
        self._swa_model = AveragedModel(model)

    def train_epoch(self, dataloader):
        self._model.train()
        total_loss = 0.0
        batches = 0

        for X_batch, y_batch in dataloader:
            predictions = self._model(X_batch)
            loss = self._criterion(predictions, y_batch)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            batches += 1

        if batches > 0:
            return total_loss / batches
        return 0.0

    def validate_epoch(self, dataloader):
        self._model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                predictions = self._model(X_batch)
                loss = self._criterion(predictions, y_batch)
                total_loss += loss.item()
                batches += 1

        if batches > 0:
            return total_loss / batches
        return 0.0

    def fit(self, train_loader, val_loader, epochs, patience, checkpoint_path):
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        swa_updates = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self._scheduler.step(val_loss)
            current_lr = self._optimizer.param_groups[0]['lr']

            in_swa_window = epoch >= self._swa_start and (self._swa_end is None or epoch < self._swa_end)
            if in_swa_window:
                self._swa_model.update_parameters(self._model)
                swa_updates += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(self._model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            # print(f"Epoch {epoch + 1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: epoch {best_epoch} ({best_val_loss:.4f}) | LR: {current_lr:.2e} | SWA updates: {swa_updates}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} - no improvement for {patience} epochs")
                break

        if swa_updates > 0:
            print(f"Applying SWA weights averaged over {swa_updates} updates (starting epoch {self._swa_start + 1}).")
            self._model.load_state_dict(self._swa_model.module.state_dict())
        else:
            self._model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            print(f"Restored best model from epoch {best_epoch} (val loss: {best_val_loss:.4f})")

        return train_losses, val_losses