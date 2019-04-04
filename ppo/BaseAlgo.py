import torch
import torch.nn as nn
from torch import optim
import random, csv, copy, os
import numpy as np
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

class BaseAlgo:
    def __init__(self, ENV, model, make_env, N, T, VecEnv=SubprocVecEnv, storage_size=int(1e4), optimizer=optim.Adam, ep_len=int(1e3), N_ch=4,
                n_mid=512, batch_size=32, gamma=0.99, max_grad_norm=0.5, lr=7e-4, lambda_gae=0.98, device=None, cuda_id=0, seed=0, rec_times=200, *args, **kwargs):
                #optimizer=partial(torch.optim.RMSprop, eps=1e-5, alpha=0.99, momentum=0.95), 
        if N == 1:
            VecEnv = DummyVecEnv
        self.reset_seed(seed=seed)
        self.envs = VecEnv([make_env(ENV) for _ in range(N)])
        self.ac_s = self.envs.action_space.n
        self.ob_s = self.envs.observation_space.shape
        self.model = model(self.ob_s, self.ac_s, n_mid=n_mid)
        self.storage_size = storage_size
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.N = N
        self.T = T
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.rec_times = rec_times
        self.max_grad_norm = max_grad_norm
        self.ep_len = ep_len
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
        
    def __call__(self, path="save_path/", name=""):
        self.mkdir(path, name)
        self.storage.states[0] = torch.from_numpy(self.envs.reset())
        scores = np.zeros(self.rec_times + 1)
        rec_interval = self.T // self.rec_times
        rec_ind = 0
        for t in range(self.T):
            self.one_step(t)
            self.update(t)
            if (t + 1) == self.ep_len:
                scores[rec_ind] = self.storage.rewards[:self.ep_len].sum() / self.N
                print(t + 1, scores[rec_ind])
                rec_ind += 1
            if (t + 1) % rec_interval == 0:
                ind = np.arange(t + 1 - self.ep_len, t + 1) % self.storage_size
                scores[rec_ind] = self.storage.rewards[ind].sum() / self.N
                print(t + 1, scores[rec_ind])
                rec_ind += 1
                torch.save(self.model.to("cpu").state_dict(), path+name+"/"+str(t+1)+".pth")
                self.model.to(self.device)
        with open(path+"score_csv/"+name+".csv", "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(scores)
        plt.plot(np.arange(self.rec_times+1) * rec_interval, scores, label="score")
        #plt.legend()
        plt.xlabel("training times")
        plt.ylabel("average score per episodes")
        plt.savefig(path+"score/"+name+".png")
        plt.close()
    
    def mkdir(self, path, name):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        try:
            os.mkdir(path+"score")
        except FileExistsError:
            pass
        try:
            os.mkdir(path+"score_csv")
        except FileExistsError:
            pass
        try:
            os.mkdir(path+name)
        except FileExistsError:
            pass


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
    
    def insert(self, state, done, reward, action, t):
        if not hasattr(state, "numpy"):
            state = torch.from_numpy(state)
        self.states[t+1] = state
        self.masks[t][done] = 0.0
        self.rewards[t] = reward
        self.actions[t] = action
