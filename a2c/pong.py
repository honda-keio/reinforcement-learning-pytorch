import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gym, os, argparse, csv
from datetime import datetime, timedelta, timezone
from functools import partial
from nn_model import CnnModel
from a2c import AAC
from wrappers import make_env as make_env_
from argments import get_args

class RewEnv(gym.RewardWrapper):
    def __init__(self, env):
        self.point = [0,0]
        return super().__init__(env)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward == 1:
            done = True
            self.point[0] += 1
        elif reward == -1:
            done = True
            self.point[1] += 1
        return obs, reward, done, info
    def reset(self):
        if self.point[0] == 20 or self.point[1] == 20:
            self.point = [0,0]
            return self.env.reset()
        elif self.point[0] == 0 or self.point[1] == 0:
            return self.env.reset()
        else:            
            obs, _, _ , _ = self.env.step(0)
            return obs

def make_env(ENV):
    def _x():
        return RewEnv(make_env_(ENV)())
    return _x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)
    parser.add_argument("--path", default="pong/")
    args = parser.parse_args()
    ENV = "PongNoFrameskip-v4"
    max_epochs = args.max_epochs
    gamma = args.gamma
    n_mid = 512
    N = args.N
    T = args.T
    max_grad_norm = args.max_grad_norm
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    v_coef = args.v_coef
    lr = args.lr
    path = args.path
    name = "lr" + str(lr)
    name += "v_coef" + str(v_coef)
    JST = timezone(timedelta(hours=+9), "JST")
    print(datetime.now(JST), name)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    try:
        os.mkdir(path+"reward")
    except FileExistsError:
        pass
    try:
        os.mkdir(path+"reward_csv")
    except FileExistsError:
        pass
    try:
        os.mkdir(path+name)
    except FileExistsError:
        pass
    
    a2c = AAC(ENV, CnnModel, max_epochs, N, T, make_env, optimizer=partial(torch.optim.RMSprop, eps=1e-5, alpha=0.99), lr=lr, max_grad_norm=max_grad_norm, 
        n_mid=n_mid, v_coef=v_coef, device=device)
    rs = a2c(path+name+"/", 500)
    with open(path+"reward_csv/"+name+".csv", "a") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(rs)
    plt.plot(range(len(rs)), rs)
    plt.savefig(path+"reward/"+name+".png")
    plt.close()
    print(datetime.now(JST))