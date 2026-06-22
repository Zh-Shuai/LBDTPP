import torch
import pickle
import numpy as np
from abc import ABC
from prettytable import PrettyTable
from tpp_experiment.metrics import get_distances_diffusion, type_rmse_diffusion, time_rmse_tensor, sMape_tensor

import os
import time
import pathlib

HOME = str(pathlib.Path.home())


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def find_index(type_seqs, padding_type):
    b, s = type_seqs.shape

    mask = type_seqs == padding_type
    first_true_idx = torch.argmax(mask.int(), dim=1)

    all_false_mask = ~mask.any(dim=1)
    first_true_idx[all_false_mask] = s

    result_idx = first_true_idx - 1
    result_idx = torch.clamp(result_idx, min=0)

    return result_idx


def sequence_truncation(seqs, exceeded_time_index):
    b = seqs.shape[0]
    new_seqs = []

    for i in range(b):
        idx = exceeded_time_index[i].item()
        trunc_seq = seqs[i, :idx + 1]
        new_seqs.append(trunc_seq)

    return new_seqs


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class BaseExperiment(object):
    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch,
                 log_path, eval_every, check_every, ema=None):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics) == 0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics) == 0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def create_folders(self):
        # Create log folder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None:
            if not os.path.exists(self.check_path):
                os.makedirs(self.check_path)
                print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):
        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path, 'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def checkpoint_save(self, name='checkpoint.pt'):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, name))

    def checkpoint_load(self, check_path, name='checkpoint.pt'):
        checkpoint = torch.load(os.path.join(check_path, name))
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def run(self, epochs):
        best_eval = 9999999999999999999
        saved_epoch = 0

        for epoch in range(self.current_epoch, epochs):

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # Eval
            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
            else:
                eval_dict = None

            # Checkpoint
            self.current_epoch += 1
            
            if (epoch + 1) % self.eval_every == 0:
                if (self.args.task == 'unconditional' and self.args.dataset in ['taobao']) or \
                    (self.args.task == 'conditional' and self.args.dataset in ['so', 'retweet']):
                    eval_loss = eval_dict['rmse_type_wo_filter']
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        self.checkpoint_save()
                        print('We will save the model at {}'.format(epoch + 1))
                        saved_epoch = epoch + 1
                    else:
                        print('We still use the model at {}'.format(saved_epoch))
                else:
                    eval_loss = eval_dict['otd_wo_filter']
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        self.checkpoint_save()
                        print('We will save the model at {}'.format(epoch + 1))
                        saved_epoch = epoch + 1
                    else:
                        print('We still use the model at {}'.format(saved_epoch))



class DiffusionExperiment(BaseExperiment, ABC):
    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):
        if args.log_home is None:
            self.log_base = os.path.join(HOME, 'log')
        else:
            self.log_base = os.path.join(args.log_home, 'log')

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.eval_every
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        args.log_path = os.path.join(self.log_base, data_id, model_id, optim_id, args.name)

        # Move model
        model = model.to(args.device)

        # Create EMA model
        ema = EMA(0.9)
        ema.register(model)

        # Init parent
        super(DiffusionExperiment, self).__init__(model=model, ema=ema,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, data_id, model_id, optim_id,
                                                                        args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every)

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def run(self):
        super(DiffusionExperiment, self).run(epochs=self.args.epochs)



class Experiment(DiffusionExperiment):
    def train_fn(self, epoch):

        self.model.train()
        loss_sum = 0.0
        diff_loss_sum = 0.0
        time_loss_sum = 0.0
        type_loss_sum = 0.0
        loss_count = 0
        loss_moving = None

        for iteration, batch in enumerate(self.train_loader):
            if self.args.task == 'conditional':
                hist_e = batch.history_types.long()
                hist_x = batch.history_dt
                tgt_e = batch.target_types.long()
                tgt_x = batch.target_dt
                loss, diff_loss, time_loss, type_loss = self.model(hist_x, hist_e, tgt_x, tgt_e, task=self.args.task)
            elif self.args.task == 'unconditional':
                x = batch.normed_time_delta_seq
                e = batch.event_seq.long()
                loss, diff_loss, time_loss, type_loss = self.model(x, e, task=self.args.task)
            else:
                raise NotImplementedError(f'Unknown task: {self.args.task}')

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

            self.ema.update(self.model)

            if (iteration + 1) % self.args.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler_epoch.step()

            if self.args.task == 'conditional':
                seq_nums = len(hist_x)
            else:
                seq_nums = len(x)

            loss_sum += loss.detach().cpu().item() * seq_nums
            diff_loss_sum += diff_loss.detach().cpu().item() * seq_nums
            time_loss_sum += time_loss.detach().cpu().item() * seq_nums
            type_loss_sum += type_loss.detach().cpu().item() * seq_nums
            loss_count += seq_nums

            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            if self.args.debug and loss_count > self.args.debug:
                break
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Total loss : {:.3f}, '
                  'Total diff loss: {:.3f}, Total time loss: {:.3f}, Total type loss: {:.3f}'.format(epoch + 1,
                                                                                                     self.args.epochs,
                                                                                                     loss_count,
                                                                                                     len(self.train_loader.dataset),
                                                                                                     loss_sum / loss_count,
                                                                                                     diff_loss_sum / loss_count,
                                                                                                     time_loss_sum / loss_count,
                                                                                                     type_loss_sum / loss_count), end='\r')
        print('')

        return {'total_loss': loss_sum / loss_count, 'diff_loss': diff_loss_sum / loss_count,
                'time_loss': time_loss_sum / loss_count, 'type_loss': type_loss_sum / loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():

            total_distances_wo_filter = []
            total_rmse_types_wo_filter = []

            pred_e_total = torch.empty(0, self.args.tgt_len).to('cpu')
            pred_x_total = torch.empty(0, self.args.tgt_len).to('cpu')
            gt_e_total = torch.empty(0, self.args.tgt_len).to('cpu')
            gt_x_total = torch.empty(0, self.args.tgt_len).to('cpu')


            for iteration, batch in enumerate(self.eval_loader):
                if self.args.task == 'conditional':
                    hist_x = batch.history_dt
                    hist_e = batch.history_types.long()
                    tgt_x = batch.target_dt * self.args.t_scale
                    tgt_e = batch.target_types.long()
                    gt_x = tgt_x.cpu()
                    gt_e = tgt_e.cpu().long()
                    pred_x, pred_e = self.model.conditional_generation_sample(hist_x, hist_e, self.args.tgt_len, hist_x.shape[0])
                elif self.args.task == 'unconditional':
                    x = batch.normed_time_delta_seq
                    e = batch.event_seq.long()
                    index = find_index(e, self.args.num_classes)
                    x = sequence_truncation(x, index)
                    e = sequence_truncation(e, index)
                    stop_time = batch.normed_final_time
                    gt_e = [tensor.cpu().long() for tensor in e]
                    gt_x = [tensor.cpu()*self.args.t_scale for tensor in x]
                    pred_x, pred_e = self.model.unconditional_generation_sample(stop_time, stop_time.shape[0])
                else:
                    raise NotImplementedError(f'Unknown task: {self.args.task}')

                pred_e = [tensor.cpu().long() for tensor in pred_e]
                pred_x = [tensor.cpu()*self.args.t_scale for tensor in pred_x]

                if self.args.task == 'conditional':
                    pred_e_total = torch.cat([pred_e_total, torch.stack(pred_e)], dim=0)
                    pred_x_total = torch.cat([pred_x_total, torch.stack(pred_x)], dim=0)
                    gt_e_total = torch.cat([gt_e_total, gt_e], dim=0)
                    gt_x_total = torch.cat([gt_x_total, gt_x], dim=0)

                filter = False
                distances_wo_filter = get_distances_diffusion(pred_x, pred_e, gt_x, gt_e, self.args.num_classes, filter,
                                                              None, self.args.distance_del_cost, self.args.trans_cost)
                total_distances_wo_filter += list(np.array(distances_wo_filter))

                filter = False
                rmse_types_wo_filter = type_rmse_diffusion(pred_x, pred_e, gt_x, gt_e, self.args.num_classes, filter, None)
                total_rmse_types_wo_filter += list(np.array(rmse_types_wo_filter))


            total_distances_wo_filter = np.mean(total_distances_wo_filter)
            total_rmse_types_wo_filter = np.mean(total_rmse_types_wo_filter)

            print('Evaluating train for N={} forecasting. Epoch: {}/{}, '
                  'OTD fixed forecasting: {:.3f}, rmse_type fixed forecasting: {:.3f}'
                  .format(self.args.tgt_len, epoch + 1, self.args.epochs, total_distances_wo_filter, total_rmse_types_wo_filter), end='\r')
            print('')
            
            if self.args.task == 'conditional':
                rmse_mean, _ = time_rmse_tensor(pred_x_total, gt_x_total)
                print('rmse time is {:.3f}'.format(rmse_mean))

                smape_mean, _ = sMape_tensor(pred_x_total, gt_x_total)
                print('sMAPE is {:.3f}'.format(smape_mean))
            
        return {'otd_wo_filter': total_distances_wo_filter,
                'rmse_type_wo_filter': total_rmse_types_wo_filter}