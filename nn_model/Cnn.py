import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_model import BaseModel

class CnnModel(BaseModel):
    def __init__(self, ob_s, ac_s, n_mid=512):
        super().__init__()
        def init(module, gain=nn.init.calculate_gain("relu")):
            #nn.init.orthogonal_(module.weight.data, gain)
            #nn.init.constant_(module.bias.data, 0)
            return module
        self.cnn_layer = nn.Sequential(
            init(nn.Conv2d(ob_s[0], 32, 8, stride=4)),
            nn.ReLU(),
            init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.conv_out = self._get_conv_out(ob_s)
        self.L = nn.Sequential(
            nn.Linear(self.conv_out, n_mid),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            #nn.Linear(self.conv_out, n_mid),
            #nn.ReLU(),
            #nn.Linear(n_mid, ac_s),
            nn.Linear(n_mid, ac_s)
        )
        self.critic = nn.Sequential(
            #nn.Linear(self.conv_out, n_mid),
            #nn.ReLU(),
            #nn.Linear(n_mid, 1)
            nn.Linear(n_mid, 1)
            )

    def _get_conv_out(self, ob_s):
        x = torch.zeros(1, *ob_s)
        h = self.cnn_layer(x)
        return int(np.prod(h.size()))

    def phi(self, x):
        h = self.cnn_layer(x).view(-1, self.conv_out)
        return self.L(h)