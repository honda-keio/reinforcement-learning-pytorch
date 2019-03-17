def get_args(parser):
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lambda_gae", type=float, default=0.98)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("-e", "--max_epochs", type=int, default=int(1e6))
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--v_coef", type=float, default=0.5)
    

