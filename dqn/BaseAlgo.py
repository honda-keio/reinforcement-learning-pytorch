import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gym
import random, time
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class BaseAlgo:
    def __init__(self, ENV, model, max_epochs, N, T, make_env, t_step=1, optimizer=optim.Adam, n_mid=512, 
                gamma=0.99, lambda_gae=0.98, v_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, device=None):
        self.envs = SubprocVecEnv([make_env(ENV) for _ in range(N)])
        self.ob_s = self.envs.observation_space.shape
        self.t_step = t_step
        ac_s = self.envs.action_space.n
        self.model = model(self.ob_s, ac_s, n_mid=n_mid)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        self.N = N
        self.T = T
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.v_coef = v_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self._init_strage()

    def reset_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _init_strage(self):
        self.states = torch.zeros([self.T+1, self.N, *self.ob_s])
        self.masks = np.ones([self.T, self.N, 1], dtype=np.float32)
        self.rewards = np.zeros([self.T, self.N], dtype=np.float32)
        self.actions = np.zeros([self.T, self.N], dtype=np.int32)
    
    def insert(self, state, done, reward, action, t):
        if not hasattr(state, "numpy"):
            state = torch.from_numpy(state)
        self.states[t+1] = state
        self.masks[t][done] = 0.0
        self.rewards[t] = reward
        self.actions[t] = action

    def one_step(self, t):
        with torch.no_grad():
            action = self.model.act(x=self.states[t].to(self.device)).to("cpu").view(-1).numpy()
            state, reward, done, _ = self.envs.step(action)
            self.insert(state, done, reward, action, t)
