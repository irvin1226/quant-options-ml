import torch
import pandas as pd
import numpy as np
from data_utils import Features

# Loads all files into GPU VRAM once at startup so each epoch runs entirely on GPU.
class VRAMDataLoader:

    def __init__(self, file_list: list, features: Features, batch_size: int, shuffle: bool, device: str):
        X_tensors = []
        y_tensors = []

        # Load one file at a time to avoid RAM exhaustion
        for path in file_list:
            df = pd.read_parquet(path)
            X_np, cleaned = features.transform(df)
            y_np = cleaned['label'].values
            del df, cleaned

            X_t = torch.tensor(X_np, dtype=torch.float).to(device)
            y_t = torch.tensor(y_np, dtype=torch.float).reshape(-1, 1).to(device)
            del X_np, y_np

            X_tensors.append(X_t)
            y_tensors.append(y_t)

        self.X = torch.cat(X_tensors, dim=0)
        self.y = torch.cat(y_tensors, dim=0)
        del X_tensors, y_tensors

        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.dataset_len = self.X.shape[0]

    def __iter__(self):
        # Shuffle indices on GPU so no data ever touches the CPU during training
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.X.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        if self.indices is not None:
            idx     = self.indices[self.i : self.i + self.batch_size]
            batch_X = self.X[idx]
            batch_y = self.y[idx]
        else:
            batch_X = self.X[self.i : self.i + self.batch_size]
            batch_y = self.y[self.i : self.i + self.batch_size]

        self.i += self.batch_size
        return batch_X, batch_y

    # Returns total number of batches, rounding up for the final partial batch
    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


# Loads exit label parquets into VRAM.
class ExitVRAMDataLoader:

    def __init__(self, file_list: list, exit_columns: list, scaler, batch_size: int, shuffle: bool, device: str, label_col: str = 'remaining_upside'):
        X_tensors = []
        y_tensors = []

        for path in file_list:
            df = pd.read_parquet(path)

            for col in exit_columns:
                df[col] = df[col].fillna(0.0)
            df[label_col] = df[label_col].fillna(0.0)

            X_raw = df[exit_columns].values.astype(np.float32)
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
            X_np = scaler.transform(X_raw)
            y_np = df[label_col].values.astype(np.float32)
            del df

            X_t = torch.tensor(X_np, dtype=torch.float).to(device)
            y_t = torch.tensor(y_np, dtype=torch.float).reshape(-1, 1).to(device)
            del X_np, y_np

            X_tensors.append(X_t)
            y_tensors.append(y_t)

        self.X = torch.cat(X_tensors, dim=0)
        self.y = torch.cat(y_tensors, dim=0)
        del X_tensors, y_tensors

        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.dataset_len = self.X.shape[0]

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.X.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        if self.indices is not None:
            idx     = self.indices[self.i : self.i + self.batch_size]
            batch_X = self.X[idx]
            batch_y = self.y[idx]
        else:
            batch_X = self.X[self.i : self.i + self.batch_size]
            batch_y = self.y[self.i : self.i + self.batch_size]

        self.i += self.batch_size
        return batch_X, batch_y

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size