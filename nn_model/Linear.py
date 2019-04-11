from torch import nn
from nn_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self, ob_s, ac_s, n_mid=10):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(*ob_s, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
        )

        self.actor = nn.Sequential(
            #nn.Linear(n_mid, n_mid),
            #nn.ReLU(),
            nn.Linear(n_mid, ac_s)
        )
        self.critic = nn.Sequential(
            #nn.Linear(n_mid, n_mid),
            #nn.ReLU(),
            nn.Linear(n_mid, 1)
        )
