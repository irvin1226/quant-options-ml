import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import numpy as np


class ModelNN(nn.Module):

    def __init__(self, input_size: int):
        super(ModelNN, self).__init__()
        self._hidden1 = nn.Linear(input_size, 256)
        self._bn1 = nn.BatchNorm1d(256)

        self._hidden2 = nn.Linear(256, 128)
        self._bn2 = nn.BatchNorm1d(128)

        self._hidden3 = nn.Linear(128, 64)
        self._bn3 = nn.BatchNorm1d(64)

        self._output = nn.Linear(64, 1)

        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self._hidden1(x)
        x = self._bn1(x)
        x = self._relu(x)
        x = self._dropout(x)

        x = self._hidden2(x)
        x = self._bn2(x)
        x = self._relu(x)
        x = self._dropout(x)

        x = self._hidden3(x)
        x = self._bn3(x)
        x = self._relu(x)
        x = self._dropout(x)

        x = self._output(x)
        return x


class ModelGBT:

    def __init__(self, n_estimators: int, learning_rate: float, max_depth: int, scale_pos_weight: int, random_state: int = 42):
        self._model = lgb.LGBMClassifier(
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth,
            objective = 'binary',
            scale_pos_weight = scale_pos_weight,
            force_row_wise = True,
            n_jobs = -1,
            random_state = random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)

    def predict(self, X: np.ndarray):
        # predict_proba returns [prob_negative, prob_positive]. Column 1 is the buy probability
        return self._model.predict_proba(X)[:, 1]


class Trainer:

    def __init__(self, model: ModelNN, device: str, learning_rate: float, pos_weight: float):
        self._device = device
        self._model = model.to(device)
        # pos_weight must be a tensor on the same device as the model
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self._optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

    def fit(self, train_loader, val_loader, epochs: int, patience: int, checkpoint_path: str):
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(self._model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: epoch {best_epoch} ({best_val_loss:.4f})")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} - no improvement for {patience} epochs")
                break

        # Restore the best checkpoint rather than the final (potentially overfit) weights
        self._model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Restored best model from epoch {best_epoch} (val loss: {best_val_loss:.4f})")

        return train_losses, val_losses