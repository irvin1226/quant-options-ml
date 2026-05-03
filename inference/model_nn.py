import torch
import torch.nn as nn


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