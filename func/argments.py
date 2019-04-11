def get_args(parser):
    parser.add_argument("--T", type=float, default=5e5)
    parser.add_argument("--storage_size", type=float, default=5e4)
    parser.add_argument("--targ_synch", type=float, default=1e3)
    parser.add_argument("--ep_len", type=float, default=1e3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--update_per_ep", type=int, default=1, help="for ppo: K times update using same data")
    parser.add_argument("--rec_times", type=float, default=100, help="record frequency")
    parser.add_argument("--N", type=int, default=16, help="num of parallel envs")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lambda_gae", type=float, default=0.98)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--N_step", type=int, default=5, help="for a2c")
    parser.add_argument("--v_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", default="ppo", help="which algorithm do you use? ppo or dqn")
    

def arg2kwargs(args):
    kwargs = {
        "T": int(args.T),
        "storage_size": int(args.storage_size),
        "batch_size": args.batch_size,
        "update_per_ep": args.update_per_ep,
        "targ_synch": int(args.targ_synch),
        "ep_len": int(args.ep_len),
        "N": args.N,
        "lr": args.lr,
        "rec_times": int(args.rec_times),
        "gamma": args.gamma,
        "lambda_gae": args.lambda_gae,
        "max_grad_norm": args.max_grad_norm,
        "no_cuda": args.no_cuda,
        "N_step": args.N_step,
        "v_coef": args.v_coef,
        "ent_coef":  args.ent_coef,
        "seed": args.seed
    }
    return kwargs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    get_args(parser)
    args = parser.parse_args()
    print(arg2kwargs(args))