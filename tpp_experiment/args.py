def add_exp_args(parser):
    # Train params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--need_regularization', action='store_true')
    parser.add_argument('--task', type=str, default='conditional', choices={'conditional', 'unconditional'})

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=2)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)

    # Eval params
    parser.add_argument('--distance_del_cost', type=list, default=[0.05, 0.5, 1, 1.5, 2, 3, 4])
    parser.add_argument('--trans_cost', type=float, default=1.0)


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='taxi',
                        choices={'taxi', 'taobao', 'so', 'retweet', 'mooc', 'amazon'})
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='data/taxi/')
    parser.add_argument('--validation', type=eval, default=True)
    parser.add_argument('--tgt_len', type=int, default=20)

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=eval, default=False)


def add_model_args(parser):
    # Diffusion params
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--sampling_steps', type=int, default=50)

    # Transformer params.
    parser.add_argument('--model_length', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--transformer_dim', type=int, default=32)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--loss_lambda', type=float, default=1)
    
def add_optim_args(parser):
    # Model params
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.999)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--scheduler', type=str, default='cosanneal')
    parser.add_argument('--grad_norm', action='store_true')
    parser.add_argument('--no-grad_norm', action='store_false')


def add_eval_args(parser):
    # Eval params
    parser.add_argument('--model', type=str, default=None)
