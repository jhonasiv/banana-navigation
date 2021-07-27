import torch
from torch import nn
from torch.nn import functional as f


class DQNModel(nn.Module):
    def __init__(self, input_features, output_features, seed):
        super(DQNModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(input_features, 64), nn.Linear(64, 128), nn.Linear(128, 256),
                                   nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 64),
                                   nn.Linear(64, output_features))
    
    def forward(self, x):
        for layer in self.model[:-1]:
            x = layer(x)
            x = f.relu(x)
        x = self.model[-1](x)
        return x
