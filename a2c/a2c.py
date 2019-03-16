import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gym
import random, time
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class AAC:
    def __init__(self, ENV, model, max_epochs, N, T, make_env, optimizer=optim.Adam, n_mid=10, 
                gamma=0.99, lambda_gae=0.98, v_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=0.01, device=None):
        self.envs = SubprocVecEnv([make_env(ENV) for _ in range(N)])
        self.ob_s = self.envs.observation_space.shape
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

    def reset_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def T_step(self, state):
        states = torch.zeros([self.T+1, self.N, *self.ob_s])
        masks = np.ones([self.T, self.N, 1], dtype=np.float32)
        rewards = np.zeros([self.T, self.N], dtype=np.float32)
        actions = np.zeros([self.T, self.N], dtype=np.int32)
        states[0] = state
        with torch.no_grad():
            for t in range(self.T):
                actions[t] = self.model.act(x=states[t].to(self.device)).to("cpu").view(-1).numpy()
                state, rewards[t], done, _ = self.envs.step(actions[t])
                states[t+1] = torch.from_numpy(state)
                masks[t][done] = 0.0
        return states, masks, rewards, actions, 
    def calc_returns(self, rewards, masks, states):
        """
        with torch.no_grad():
            last_v  = self.model.V(last_s.to(self.device)).to("cpu")
        rewards = torch.from_numpy(np.expand_dims(rewards, -1))
        masks = torch.from_numpy(masks)
        returns = torch.zeros([self.T, self.N, 1])
        returns[-1] = rewards[-1] + self.gamma * masks[-1] * last_v
        for t in reversed(range(self.T-1)):
            returns[t] = rewards[t] + self.gamma * masks[t] * returns[t+1]            
        return returns
        """
        returns = torch.zeros([self.T+1, self.N, 1])
        with torch.no_grad():
            V = self.model.V(states[-1].to(self.device)).to("cpu")
            for t in reversed(range(self.T)):
                delta_t = rewards[t] + self.gamma * V
                V = self.model.V(states[t].to(self.device)).to("cpu")
                delta_t -= V
                returns[t] = delta_t + masks[t] * self.gamma * self.lambda_gae * returns[t+1]
        return returns[:-1]

    def update(self, states, actions, returns):
        actions = torch.from_numpy(actions).view(-1,1).long().to(self.device)
        pi, V = self.model(states[:-1].view(-1, *self.ob_s).to(self.device))
        pi_log_prob = F.log_softmax(pi, dim=1)
        ADV = returns.view(-1, 1).to(self.device) - V
        pi_loss = - pi_log_prob.gather(1, actions) * ADV.detach()
        V_loss = ADV.pow(2)
        entropy_loss = -(pi_log_prob * pi_log_prob.exp()).sum(-1)
        total_loss = pi_loss.mean() + self.v_coef * V_loss.mean() + self.ent_coef * entropy_loss.mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def __call__(self, name="a2c", r_interval=200, seed=0):
        start = time.time()
        self.reset_seed(seed)
        state = torch.from_numpy(self.envs.reset())
        rs = np.zeros((self.max_epochs * self.T) // r_interval)
        i = 0
        for epoch in range(self.max_epochs):
            states, masks, rewards, actions = self.T_step(state)
            state = states[-1]
            returns = self.calc_returns(rewards, masks, state)
            self.update(states, actions, returns)
            rs[i] += rewards.sum()
    
            if (epoch + 1) * self.T % r_interval == 0:
                sec = int(time.time() - start)
                h = sec // 3600
                sec = sec % 3600
                m = int(sec // 60)
                sec = int(sec % 60)
                print(h, m, sec, sep=":")
                rs[i] /= self.N
                print(epoch + 1, rs[i])
                torch.save(self.model.to("cpu").state_dict(), name+str(i)+".pth")
                self.model.to(self.device)
                i += 1
        torch.save(self.model.to("cpu").state_dict(), name+"last.pth")
        return rs