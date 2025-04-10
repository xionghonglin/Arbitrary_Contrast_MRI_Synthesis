from torch import dropout
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_dim=0, out_dim=0, hidden_list = [], drop=0):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            # layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)

     
