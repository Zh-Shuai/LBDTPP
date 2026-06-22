from torch.utils.data import DataLoader

from dataloader.dataset import create_collate_cond, create_collate_cond_test, create_collate_uncond, load_dataset

dataset_choices = {'taxi', 'taobao', 'so', 'retweet', 'mooc', 'amazon'}


def get_data_id(args):
    return args.dataset


def get_data_cond(args):
    assert args.dataset in dataset_choices
    print('loading {} cond datasets...'.format('train'))
    train = load_dataset(dataset_dir=args.dataset_dir, mode='train', device=args.device, data_name=args.dataset, target_length=args.tgt_len, task='conditional')
    t_scale = train.t_scale
    args.t_scale = t_scale

    print('loading {} cond datasets...'.format('val'))
    valid = load_dataset(dataset_dir=args.dataset_dir, mode='dev', t_scale=t_scale, device=args.device, data_name=args.dataset, target_length=args.tgt_len, task='conditional')

    print('loading {} cond datasets...'.format('test'))
    test = load_dataset(dataset_dir=args.dataset_dir, mode='test', t_scale=t_scale, device=args.device, data_name=args.dataset, target_length=args.tgt_len, task='conditional')


    if args.dataset == 'taxi':
        args.num_classes = 10

    if args.dataset == 'taobao':
        args.num_classes = 17

    if args.dataset == 'so':
        args.num_classes = 22

    if args.dataset == 'retweet':
        args.num_classes = 3

    if args.dataset == 'mooc':
        args.num_classes = 50

    if args.dataset == 'amazon':
        args.num_classes = 16
    

    # Data Loader
    collate_cond = create_collate_cond(args.block_size)
    collate_cond_test = create_collate_cond_test(args.block_size)

    if args.validation:
        train_loader = DataLoader(train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collate_cond)
        eval_loader = DataLoader(valid, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_cond_test)
    else:
        train_loader = None
        eval_loader = DataLoader(test, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_cond_test)

    num_classes = args.num_classes
    return train_loader, eval_loader, num_classes


def get_data_uncond(args):
    assert args.dataset in dataset_choices
    print('loading {} uncond datasets...'.format('train'))
    train = load_dataset(dataset_dir=args.dataset_dir, mode='train', device=args.device, data_name=args.dataset, task='unconditional')
    t_scale = train.t_scale
    args.t_scale = t_scale

    print('loading {} uncond datasets...'.format('val'))
    valid = load_dataset(dataset_dir=args.dataset_dir, mode='dev', t_scale=t_scale, device=args.device, data_name=args.dataset, task='unconditional')

    print('loading {} uncond datasets...'.format('test'))
    test = load_dataset(dataset_dir=args.dataset_dir, mode='test', t_scale=t_scale, device=args.device, data_name=args.dataset, task='unconditional')

    
    if args.dataset == 'taxi':
        args.num_classes = 10

    if args.dataset == 'taobao':
        args.num_classes = 17

    if args.dataset == 'so':
        args.num_classes = 22

    if args.dataset == 'retweet':
        args.num_classes = 3

    if args.dataset == 'mooc':
        args.num_classes = 50

    if args.dataset == 'amazon':
        args.num_classes = 16


    # Data Loader
    collate_uncond = create_collate_uncond(args.block_size, args)

    if args.validation:
        train_loader = DataLoader(train, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collate_uncond)
        eval_loader = DataLoader(valid, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_uncond)
    else:
        train_loader = None
        eval_loader = DataLoader(test, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_uncond)

    num_classes = args.num_classes
    return train_loader, eval_loader, num_classes
