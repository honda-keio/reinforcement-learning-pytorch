import torch, gym, argparse
from datetime import datetime, timedelta, timezone
from nn_model import LinearModel
from ppo import PPO
from argments import get_args, arg2kwargs

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
    kwargs = arg2kwargs(args)
    ENV = "CartPole-v0"
    kwargs["n_mid"] = args.n_mid
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path = args.path
    JST = timezone(timedelta(hours=+9), "JST")
    name = "lr" + str(kwargs["lr"])
    print(datetime.now(JST), name)
    
    ppo = PPO(ENV=ENV, model=LinearModel, make_env=make_env, **kwargs, device=device)
    ppo(path, name)
    
    print(datetime.now(JST))