def get_args(parser):
    parser.add_argument("--T", type=int, default=200)
    #parser.add_argument("--e_len", type=int, default=200, help="length of an episode")
    #parser.add_argument("--episodes", type=int, default=200, help="")
    parser.add_argument("--N", type=int, default=16, help="num of parallel envs")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--lambda_gae", type=float, default=0.98)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("-e", "--max_epochs", type=int, default=int(1e4))
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--N_step", type=int, default=5, help="for a2c")
    parser.add_argument("--v_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    

def arg2kwargs(args):
    kwargs = {
        "T": args.T,
        "N": args.N,
        "lr": args.lr,
        "gamma": args.gamma,
        "lambda_gae": args.lambda_gae,
        "max_grad_norm": args.max_grad_norm,
        "max_epochs": args.max_epochs,
        "no_cuda": args.no_cuda,
        "N_step": args.N_step,
        "v_coef": args.v_coef,
        "ent_coef":  args.ent_coef
    }
    return kwargs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    get_args(parser)
    args = parser.parse_args()
    print(arg2kwargs(args))