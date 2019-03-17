import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, gym, os, csv, argparse
from functools import partial
from datetime import datetime, timedelta, timezone
from nn_model import LinearModel
from a2c import AAC
from argments import get_args

class CartEnv(gym.RewardWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, self.reward(reward, done), done, info

    def reward(self, reward, done):
        if done:
            reward = -1.
        return reward

def make_env(ENV):
    def env():
        env = gym.make(ENV).unwrapped
        return CartEnv(env)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)
    parser.add_argument("--path", default="cart/")
    parser.add_argument("--n_mid", type=int, default=10)
    args = parser.parse_args()
    ENV = "CartPole-v0"
    max_epochs = args.max_epochs
    gamma = args.gamma
    lambda_gae = args.lambda_gae
    n_mid = args.n_mid
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
    JST = timezone(timedelta(hours=+9), "JST")
    name = "lr" + str(lr)
    name += "v_coef" + str(v_coef)
    name += "gae" + str(lambda_gae)
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
    
    a2c = AAC(ENV, LinearModel, max_epochs, N, T, make_env, optimizer=partial(torch.optim.RMSprop, eps=1e-5, alpha=0.99), lr=lr, max_grad_norm=max_grad_norm, 
        n_mid=n_mid, v_coef=v_coef, device=device)
    rs = a2c(path+name+"/")
    with open(path+"reward_csv/"+name+".csv", "a") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(rs)
    plt.plot(range(len(rs)), rs)
    plt.savefig(path+"reward/"+name+".png")
    plt.close()
    print(datetime.now(JST))