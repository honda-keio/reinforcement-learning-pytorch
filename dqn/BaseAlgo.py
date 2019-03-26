import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class Base:
    def __init__(self, ENV, model, N, T, make_env, VecEnv=SubprocVecEnv, storage_size=int(1e4), optimizer=optim.Adam, n_mid=512, batch_size=32,
                gamma=0.99, max_grad_norm=0.5, lr=7e-4, device=None, *args, **kwargs):
        self.envs = VecEnv([make_env(ENV) for _ in range(N)])
        ac_s = self.envs.action_space.n
        self.ob_s = self.envs.observation_space.shape
        self.model = model(self.ob_s, ac_s, n_mid=n_mid)
        self.storage_size = storage_size
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.N = N
        self.T = T
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def reset_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

class BaseAlgo(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = Storage(ob_s=self.ob_s, *args, **kwargs)

    def one_step(self, t):
        with torch.no_grad():
            action = self.model.act(x=self.storage.states[t].to(self.device)).to("cpu").view(-1).numpy()
            state, reward, done, _ = self.envs.step(action)
            self.storage.insert(state, done, reward, action, t)

    def calc_loss(self, *args, **kwargs):
        pass
    
    def update(self, *args, **kwargs):
        self.optimizer.zero_grad()
        self.calc_loss(*args, **kwargs).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        

class Storage:
    def __init__(self, storage_size, N, ob_s, *args, **kwargs):
        self.storage_size = storage_size
        self.N = N
        self.ob_s = ob_s
        self._init_strage()
        
    def _init_strage(self):
        self.states = torch.zeros([self.storage_size+1, self.N, *self.ob_s])
        self.masks = np.ones([self.storage_size, self.N, 1], dtype=np.float32)
        self.rewards = np.zeros([self.storage_size, self.N], dtype=np.float32)
        self.actions = np.zeros([self.storage_size, self.N], dtype=np.int32)
    
    def insert(self, state, done, reward, action, t):
        if not hasattr(state, "numpy"):
            state = torch.from_numpy(state)
        self.states[t+1] = state
        self.masks[t][done] = 0.0
        self.rewards[t] = reward
        self.actions[t] = action
