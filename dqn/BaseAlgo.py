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
from multiprocessing import Process, Pipe

class BaseAlgo:
    def __init__(self, ENV, model, make_env, N, T, VecEnv=SubprocVecEnv, storage_size=int(1e4), N_step=5, optimizer=optim.Adam, ep_len=int(1e3), N_ch=4, 
                n_mid=512, batch_size=32, gamma=0.99, max_grad_norm=0.5, lr=7e-4, device=None, seed=0, rec_times=200, *args, **kwargs):
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
        self.N_step = N_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.rec_times = rec_times
        self.max_grad_norm = max_grad_norm
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.storage = Storage(storage_size=storage_size, N=N, ob_s=self.ob_s, *args, **kwargs)
        self.pef_ch = PefoCheck(ENV, make_env, N_ch, ep_len, self.device)

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
        
    def __call__(self, path="save_path/", name=""):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        try:
            os.mkdir(path+"score")
        except FileExistsError:
            pass
        """
        try:
            os.mkdir(path+"loss")
        except FileExistsError:
            pass
        """
        try:
            os.mkdir(path+"score_csv")
        except FileExistsError:
            pass
        try:
            os.mkdir(path+name)
        except FileExistsError:
            pass
        self.storage.states[0] = torch.from_numpy(self.envs.reset())
        scores = np.zeros(self.rec_times + 1)
        #losses = np.zeros(self.rec_times)
        rec_interval = self.T // self.rec_times
        rec_ind = 0
        torch.multiprocessing.set_start_method("spawn")
        recv_end, send_end = Pipe(False)
        #p = Process(target=self.pef_ch, args=(0, self.model, send_end))
        p = Process(target=self.pef_ch, args=(0, copy.deepcopy(self.model).to(self.device), send_end))
        p.start()
        for t in range(self.T):
            self.one_step(t)
            if (t + 1) % self.N_step == 0:
                self.update(t)
            if (t + 1) % rec_interval == 0:
                """
                index = np.arange(t + 1 - self.rec_times, t + 1) % self.storage_size
                scores[rec_ind] = self.storage.rewards[index].mean()
                losses[rec_ind] = self.storage.losses[index].mean()
                print(t + 1, "scores", scores[rec_ind], "loss", losses[rec_ind])
                """
                p.join()
                scores[rec_ind] = recv_end.recv()
                recv_end, send_end = Pipe(False)
                p = Process(target=self.pef_ch, args=(t + 1, copy.deepcopy(self.model).to(self.device), send_end))
                p.start()
                rec_ind += 1
                torch.save(self.model.to("cpu").state_dict(), path+name+"/"+str(t+1)+".pth")
                self.model.to(self.device)
        p.join()
        scores[rec_ind] = recv_end.recv()
        with open(path+"score_csv/"+name+".csv", "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(scores)
        plt.plot(np.arange(self.rec_times + 1) * rec_interval, scores, label="score")
        #plt.legend()
        plt.xlabel("training times")
        plt.ylabel("average score per episodes")
        plt.savefig(path+"score/"+name+".png")
        plt.close()
        """
        plt.plot(range(len(scores)), losses)
        plt.savefig(path+"loss/"+name+".png")
        plt.close()
        """

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


class PefoCheck:
    def __init__(self, ENV, make_env, N, T, device, VecEnv=DummyVecEnv):
        self.envs = VecEnv([make_env(ENV) for _ in range(N)])
        self.N = N
        self.T = T
        self.device = device
    def __call__(self, t, model, send_end):
        rewards = 0
        st = self.envs.reset()
        for _ in range(self.T):
            st = torch.from_numpy(st).to(self.device)
            act = model.act(st).to("cpu").view(-1).numpy()
            st, r, _, _ = self.envs.step(act)
            rewards += r.sum()
        rewards /= self.N
        print(t, "score", rewards)
        send_end.send(rewards)
