import torch
import torch.nn as nn
from torch import optim
import random, csv
import numpy as np
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class BaseAlgo:
    def __init__(self, ENV, model, make_env, N, T, VecEnv=SubprocVecEnv, storage_size=int(1e4), N_step=5, optimizer=partial(torch.optim.RMSprop, eps=1e-5, alpha=0.99), 
                n_mid=512, batch_size=32, gamma=0.99, max_grad_norm=0.5, lr=7e-4, device=None, seed=0, rec_interval=200, *args, **kwargs):
        self.reset_seed(seed=seed)
        self.envs = VecEnv([make_env(ENV) for _ in range(N)])
        self.ac_s = self.envs.action_space.n
        self.ob_s = self.envs.observation_space.shape
        self.model = model(self.ob_s, self.ac_s, n_mid=n_mid)
        self.storage_size = storage_size
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.N = N
        self.T = T
        self.N_step = N_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.rec_interval = rec_interval
        self.max_grad_norm = max_grad_norm
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.storage = Storage(storage_size=storage_size, N=N, ob_s=self.ob_s, *args, **kwargs)

    def reset_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def act(self, t):
        with torch.no_grad():
            action = self.model.act(x=self.storage.states[t].to(self.device)).to("cpu").view(-1).numpy()
        return action

    def one_step(self, t):
        t %= self.storage_size
        with torch.no_grad():
            action = self.act(t)
        state, reward, done, _ = self.envs.step(action)
        self.storage.insert(state, done, reward, action, t)

    def calc_loss(self, t, *args, **kwargs):
        raise NotImplementedError("prease implement loss function")
    
    def update(self, t, *args, **kwargs):
        self.optimizer.zero_grad()
        loss = self.calc_loss(t, *args, **kwargs)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.storage.losses[t%self.storage_size] = loss.item()
        
    def __call__(self, path="", name=""):
        self.storage.states[0] = torch.from_numpy(self.envs.reset())
        costs = np.zeros(self.T // self.rec_interval)
        losses = np.zeros(self.T // self.rec_interval)
        rec_ind = 0
        for t in range(self.T):
            self.one_step(t)
            if (t + 1) % self.N_step == 0:
                self.update(t)
            if (t + 1) % self.rec_interval == 0:
                index = np.arange(t + 1 - self.rec_interval, t + 1) % self.storage_size
                costs[rec_ind] = (self.storage.rewards[index] == -1).sum()
                losses[rec_ind] = self.storage.losses[index].mean()
                print(t + 1, "costs", costs[rec_ind], "loss", losses[rec_ind])
                rec_ind += 1
                torch.save(self.model.to("cpu").state_dict(), path+name+"/"+str(t+1)+".pth")
                self.model.to(self.device)
        with open(path+"cost_csv/"+name+".csv", "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(costs)
        plt.plot(np.arange(len(costs))*self.rec_interval, costs)
        plt.xlabel("training times")
        plt.ylabel("average cost per episodes")
        plt.savefig(path+"cost/"+name+".png")
        plt.close()
        plt.plot(range(len(costs)), losses)
        plt.savefig(path+"loss/"+name+".png")
        plt.close()

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
        self.actions = np.zeros([self.storage_size, self.N], dtype=np.int64)
        self.losses = np.zeros([self.storage_size, self.N], dtype=np.float32)
    
    def insert(self, state, done, reward, action, t):
        if not hasattr(state, "numpy"):
            state = torch.from_numpy(state)
        self.states[(t+1)%self.storage_size] = state
        self.masks[t][done] = 0.0
        self.rewards[t] = reward
        self.actions[t] = action
