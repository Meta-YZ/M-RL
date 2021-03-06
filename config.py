import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm", type=str, default='m-rl')
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_episodes", default=250, type=int)
    parser.add_argument("--num_threads", default=8, type=int)
    parser.add_argument("--gamma", default=0.99, type=int)
    parser.add_argument("--frames", type=int, default=int(45000), help="every now many frame action")
    parser.add_argument("--eps_frames", type=int, default=int(5000), help="every now many frame action")
    parser.add_argument("--min_eps", type=float, default=0.025, help="every now many frame action")
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="numpy/torch的随机种子，复现实验")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='CartPole-v0', help="specify the name of environment")
    parser.add_argument("--action_step", type=int, default=int(4), help="every now many frame action")

    # replay buffer parameters
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Max length for buffer")
    parser.add_argument("--batch_size", type=int, default=int(128), help="batch_size大小")
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")
    parser.add_argument("--n_step", type=int,
                        default=1, help="calculate reward per how many step")

    # network parameters
    parser.add_argument("--soft_update_tau", type=float,
                        default=1e-3, help="Max length for any episode")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--update_every", type=int, default=1)

    # M-DQN parameters
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--entropy_tau", type=float, default=0.03,
                        help='entroy的系数')
    parser.add_argument("--alpha", type=float, default=0.9,
                        help='munchansen系数')

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser
