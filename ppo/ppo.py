import torch
import torch.nn.functional as F
import copy
import numpy as np
from BaseAlgo import BaseAlgo

class PPO(BaseAlgo):
    def __init__(self, epsilon=0.2, ep_len=int(1e3), v_coef=0.5, update_per_ep=3, *args, **kwargs):
        kwargs["storage_size"] = ep_len
        kwargs["ep_len"] = ep_len
        self.v_coef = v_coef
        super().__init__(*args, **kwargs)
        self.update_per_ep = update_per_ep
        self.epsilon = epsilon

    def update(self, t, *args, **kwargs):
        if (t + 1) % self.ep_len == 0:
            old_model = copy.deepcopy(self.model).to(self.device)
            R, old_pi_log_prob = self.calc_R(old_model, *args, **kwargs)
            actions = torch.from_numpy(np.expand_dims(self.storage.actions, -1))
            for _ in range(self.update_per_ep):
                rand_list = torch.randperm(self.ep_len*self.N).view(-1, self.batch_size)
                rand_list0 = (rand_list % self.ep_len).tolist()
                rand_list1 = (rand_list // self.ep_len).tolist()
                for ind in zip(rand_list0, rand_list1):
                    super().update(t, R[ind], self.storage.states[ind], 
                        old_pi_log_prob[ind], actions[ind], *args, **kwargs)

    def calc_loss(self, t, R, states, old_pi_log_prob, actions, *args, **kwargs):
        pi, v = self.model(states.to(self.device))
        pi_log_prob = F.log_softmax(pi, dim=1)
        A = R.to(self.device) - v
        old_pi_log_prob = old_pi_log_prob.gather(1, actions).to(self.device)
        pi_log_prob = pi_log_prob.gather(1, actions.to(self.device))
        r = (pi_log_prob - old_pi_log_prob).clamp(max=3).exp()
        clip = r.clamp(min=1-self.epsilon, max=1+self.epsilon)
        L, _ = torch.stack([r * A.detach(), clip * A.detach()]).min(0)
        v_l = A.pow(2).mean()
        L = L.mean()
        loss = - L + self.v_coef * v_l
        return loss

    def calc_R(self, old_model, *args, **kwargs):
        rewards = torch.from_numpy(np.expand_dims(self.storage.rewards, -1))
        masks = torch.from_numpy(self.storage.masks)
        R = torch.zeros([self.ep_len+1, self.N, 1])
        old_pi_log_prob = torch.zeros([self.ep_len, self.N, self.ac_s])
        with torch.no_grad():
            V = old_model.V(self.storage.states[-1].to(self.device)).to("cpu")
            for t in reversed(range(self.ep_len)):
                delta_t = rewards[t] + self.gamma * V
                old_pi, V = self.model(self.storage.states[t].to(self.device))
                V = V.to("cpu")
                old_pi_log_prob[t] = F.log_softmax(old_pi, dim=1).to("cpu")
                delta_t -= V
                R[t] = delta_t + masks[t] * self.gamma * self.lambda_gae * R[t+1]
        return R[:-1], old_pi_log_prob