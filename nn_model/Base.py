from torch import nn
import torch.nn.functional as F

class BaseModel(nn.Module):
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

    def Q(self, x):
        phi = self.phi(x)
        return self.actor(phi)
        