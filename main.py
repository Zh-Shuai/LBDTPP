import os
import torch
import pickle
import numpy as np
from prettytable import PrettyTable
from tpp_experiment.experiment import Experiment, find_index, sequence_truncation
from tpp_experiment.utils import add_parent_path, set_seeds

# Data
add_parent_path(level=1)
from dataloader.get_data import get_data_id, get_data_cond, get_data_uncond

# Model
from model.bdtpp_model import get_model_id, get_model

# Optim
from tpp_experiment.optim import get_optim_id, get_optim

# Metric
from tpp_experiment.metrics import get_distances_diffusion, type_rmse_diffusion, time_rmse_tensor, sMape_tensor 

# args
import argparse
from tpp_experiment.args import add_exp_args, add_data_args, add_model_args, add_optim_args, add_eval_args


# args
def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def save_args(args):
    # Save args
    with open(os.path.join(args.log_path, 'args.pickle'), "wb") as f:
        pickle.dump(args, f)

    # Save args table
    args_table = get_args_table(vars(args))
    with open(os.path.join(args.log_path, 'args_table.txt'), "w") as f:
        f.write(str(args_table))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    add_exp_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_optim_args(parser)
    add_eval_args(parser)
    return parser.parse_args()


# Training
def run_train(args):
    set_seeds(args.seed)

    if args.task == 'conditional':
        train_loader, eval_loader, num_classes = get_data_cond(args)
    elif args.task == 'unconditional':
        train_loader, eval_loader, num_classes = get_data_uncond(args)
    else:
        raise ValueError('Unknown task')

    data_id = get_data_id(args)

    model = get_model(args, num_classes=num_classes)
    model_id = get_model_id(args)

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    optim_id = get_optim_id(args)

    args.validation = True
    exp = Experiment(args=args,
                     data_id=data_id,
                     model_id=model_id,
                     optim_id=optim_id,
                     train_loader=train_loader,
                     eval_loader=eval_loader,
                     model=model,
                     optimizer=optimizer,
                     scheduler_iter=scheduler_iter,
                     scheduler_epoch=scheduler_epoch)

    exp.run()

    return args


# Evaluation
def run_eval(args):
    eval_seed = 0

    # args
    if args == None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_path', type=str, default='./')
        parser.add_argument('--eval_seed', type=int, default=0)
        args = parser.parse_args()
        eval_seed = args.eval_seed

    path_args = '{}/args.pickle'.format(args.log_path)
    path_check = '{}/check/checkpoint.pt'.format(args.log_path)

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    assert args.tgt_len is not None, 'Currently, length has to be specified.'
    if eval_seed == 0:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(eval_seed)

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    args.num_timesteps = args.diffusion_steps

    distance_del_cost = [0.05, 0.5, 1, 1.5, 2, 3, 4]
    trans_cost = 1.0
    args.distance_del_cost = distance_del_cost
    args.trans_cost = trans_cost

    args.validation = False
    if args.task == 'conditional':
        train_loader, test_loader, num_classes = get_data_cond(args)
    elif args.task == 'unconditional':
        train_loader, test_loader, num_classes = get_data_uncond(args)
    else:
        raise ValueError('Unknown task')

    args.validation = True

    # model
    model = get_model(args, num_classes=num_classes)
    checkpoint = torch.load(path_check, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params: } training parameters.')

    # sampling
    device = args.device
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        total_distances_wo_filter = []
        total_rmse_types_wo_filter = []

        pred_e_total = torch.empty(0, args.tgt_len).to('cpu')
        pred_x_total = torch.empty(0, args.tgt_len).to('cpu')
        gt_e_total = torch.empty(0, args.tgt_len).to('cpu')
        gt_x_total = torch.empty(0, args.tgt_len).to('cpu')

        for iteration, batch in enumerate(test_loader):
            if args.task == 'conditional':
                hist_e = batch.history_types.long()
                hist_x = batch.history_dt
                tgt_e = batch.target_types.long()
                tgt_x = batch.target_dt * args.t_scale
                gt_e = tgt_e.cpu().long()
                gt_x = tgt_x.cpu()
                pred_x, pred_e = model.conditional_generation_sample(hist_x, hist_e, args.tgt_len, hist_x.shape[0])
            elif args.task == 'unconditional':
                x = batch.normed_time_delta_seq
                e = batch.event_seq.long()
                index = find_index(e, args.num_classes)
                x = sequence_truncation(x, index)
                e = sequence_truncation(e, index)
                stop_time = batch.normed_final_time
                gt_e = [tensor.cpu().long() for tensor in e]
                gt_x = [tensor.cpu()*args.t_scale for tensor in x]
                pred_x, pred_e = model.unconditional_generation_sample(stop_time, stop_time.shape[0])
            else:
                raise NotImplementedError(f'Unknown task: {args.task}')

            pred_e = [tensor.cpu().long() for tensor in pred_e]
            pred_x = [tensor.cpu()*args.t_scale for tensor in pred_x]

            if args.task == 'conditional':
                pred_e_total = torch.cat([pred_e_total, torch.stack(pred_e)], dim=0)
                pred_x_total = torch.cat([pred_x_total, torch.stack(pred_x)], dim=0)
                gt_e_total = torch.cat([gt_e_total, gt_e], dim=0)
                gt_x_total = torch.cat([gt_x_total, gt_x], dim=0)

            filter = False
            distances_wo_filter = get_distances_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter,
                                                          None, args.distance_del_cost, args.trans_cost)
            total_distances_wo_filter += list(np.array(distances_wo_filter))

            filter = False
            rmse_types_wo_filter = type_rmse_diffusion(pred_x, pred_e, gt_x, gt_e, args.num_classes, filter, None)
            total_rmse_types_wo_filter += list(np.array(rmse_types_wo_filter))

        
        total_distances_wo_filter = np.mean(total_distances_wo_filter)
        total_rmse_types_wo_filter = np.mean(total_rmse_types_wo_filter)


    print('distance (fixed forecasting) mean is {:.3f}'.format(
        total_distances_wo_filter)
    )

    print('rmse type (fixed forecasting) mean is {:.3f}'.format(
        total_rmse_types_wo_filter)
    )

    if args.task == 'conditional':
        rmse_mean, _ = time_rmse_tensor(pred_x_total, gt_x_total)
        print('rmse time is {:.3f}'.format(rmse_mean))

        smape_mean, _ = sMape_tensor(pred_x_total, gt_x_total)
        print('sMAPE is {:.3f}'.format(smape_mean))

    save_args(args)

    return args


if __name__ == '__main__':
    args = get_args()
    args = run_train(args)
    args = run_eval(args)