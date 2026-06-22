import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, total_epoch, last_epoch=-1):
        self.total_epoch = total_epoch
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1, (self.last_epoch / self.total_epoch)) for base_lr in self.base_lrs]


def get_optim_id(args):
    if args.scheduler == 'expdecay':
        return 'expdecay'
    elif args.scheduler == 'cosanneal':
        return 'cosanneal'
    return 'cosanneal'


def get_optim(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr), weight_decay=args.weight_decay)

    if args.warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None

    if args.scheduler == 'expdecay':
        scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cosanneal':
        scheduler_epoch = CosineAnnealingLR(optimizer,
                                      T_max=args.epochs,  # Maximum number of iterations.
                                      eta_min=1e-8)  # Minimum learning rate.
    else:
        scheduler_epoch = CosineAnnealingLR(optimizer,
                                      T_max=args.epochs,  # Maximum number of iterations.
                                      eta_min=1e-8)  # Minimum learning rate.


    return optimizer, scheduler_iter, scheduler_epoch