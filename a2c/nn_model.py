import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, ob_s, ac_s, n_mid=10):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(*ob_s, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
        )

        self.actor = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, ac_s)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, 1)
        )

    def forward(self, x):
        phi = self.phi(x)
        pi = self.actor(phi)
        v = self.critic(phi)
        return pi, v

    def act(self, x=None, pi=None):
        assert x is not None or pi is not None
        if pi is None:
            pi = self.actor(self.phi(x))
        pi = F.softmax(pi, dim=1)        
        return pi.multinomial(1)

    def V(self, x):
        phi = self.phi(x)
        return self.critic(phi)


class CnnModel(nn.Module):
    def __init__(self, ob_s, ac_s, n_mid=512):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(ob_s[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.conv_out = self._get_conv_out(ob_s)
        self.actor = nn.Sequential(
            nn.Linear(self.conv_out, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, ac_s),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.conv_out, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, 1)
            )

    def _get_conv_out(self, ob_s):
        x = torch.zeros(1, *ob_s)
        h = self.cnn_layer(x)
        return int(np.prod(h.size()))

    def forward(self, x):
        phi = self.cnn_layer(x).view(-1, self.conv_out)
        pi = self.actor(phi)
        v = self.critic(phi)
        return pi, v

    def act(self, x=None, pi=None):
        assert x is not None or pi is not None
        with torch.no_grad():
            if pi is None:
                h = self.phi(x)
                pi = self.actor(h)
            prob = F.softmax(pi, dim=1)
            return prob.multinomial(1)

    def V(self, x):
        phi = self.cnn_layer(x).view(-1, self.conv_out)
        return self.critic(phi)

