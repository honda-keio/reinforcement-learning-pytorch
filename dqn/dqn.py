import torch
import time, copy
import numpy as np
from BaseAlgo import BaseAlgo

class DQN(BaseAlgo):
    def __init__(self, epsilon=0.1, targ_synch=int(1e3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = torch.zeros(self.N) + epsilon
        self.targ_model = copy.deepcopy(self.model).to(self.device)
        self.targ_synch = targ_synch
        self.N_step = 1

    def td(self, t):
        if (t + 1) % self.targ_synch == 0:
            self.targ_model = copy.deepcopy(self.model).to(self.device)
        r_ind0 = torch.randint(0, self.storage_size if t > self.storage_size else t , [self.batch_size], dtype=torch.int64)
        r_ind1 = torch.randint(0, self.N, [self.batch_size], dtype=torch.int64)
        r_ind0_np = r_ind0.numpy()
        r_ind1_np = r_ind1.numpy()
        rewards = torch.from_numpy(self.storage.rewards[r_ind0_np,r_ind1_np]).view(self.batch_size, 1).to(self.device)
        masks  = torch.from_numpy(self.storage.masks[r_ind0_np,r_ind1_np]).view(self.batch_size, 1).to(self.device)
        actions = torch.from_numpy(self.storage.actions[r_ind0_np,r_ind1_np]).view(self.batch_size, 1).to(self.device)
        states = self.storage.states[r_ind0,r_ind1].view(self.batch_size, -1).to(self.device)
        states_n = self.storage.states[(r_ind0+1)%self.storage_size,r_ind1].view(self.batch_size, -1).to(self.device)
        with torch.no_grad():
            Q_n_max, _ = self.targ_model.Q(states_n).max(1, keepdim=True)
        R = rewards + masks * self.gamma * Q_n_max
        loss = (R - self.model.Q(states).gather(1, actions))
        return loss.pow(2).mean()
    
    def calc_loss(self, t):
        return self.td(t)

    def update(self, t, *args, **kwargs):
        if t * self.N > self.batch_size:
            kwargs["t"] = t
            super().update(*args, **kwargs)

    def act(self, t):
        with torch.no_grad():
            act = self.model.Q(self.storage.states[t].to(self.device)).to("cpu").argmax(1).numpy()
        for i, ep in enumerate(torch.bernoulli(self.epsilon)):
            if ep:
                act[i] = np.random.randint(0, self.ac_s, 1)
        return act

