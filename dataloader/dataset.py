import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils


class CondSeqDataset(Dataset):
    def __init__(self, dataset_dir, mode, target_length=20, t_scale=1, device=None, data_name='taxi'):
        assert mode in {'train', 'dev', 'test'}
        print(f'Loading {mode} dataset for {data_name}')

        if data_name == 'mooc':
            dataset_dir = os.path.join(dataset_dir, '{}.pkl'.format(mode))
            self.data, self.num_types = self.load_dataset_mooc(dataset_dir)
        else:
            dataset_dir = os.path.join(dataset_dir, '{}.pkl'.format(mode))
            self.data, self.num_types = self.load_dataset_hypro_format(dataset_dir, mode)

        self.target_length = target_length

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # time step
        time_seq = [[x["time_since_start"] for x in seq] for seq in self.data]
        self.time_seq = [torch.tensor(seq[1:]) for seq in time_seq]

        # event type
        event_seq = [[x["type_event"] for x in seq] for seq in self.data]
        self.event_seq = [torch.tensor(seq[1:]) for seq in event_seq]

        # inter-arrival time
        time_delta_seq = [[x["time_since_last_event"] for x in seq] for seq in self.data]
        self.unnormed_time_delta_seq = [torch.tensor(seq[1:]) for seq in time_delta_seq]

        self.t_scale = t_scale

        self.normed_time_delta_seq = [seq / self.t_scale for seq in self.unnormed_time_delta_seq]

        self.history_times = []
        self.target_times = []
        self.history_types = []
        self.target_types = []
        self.history_dt = []
        self.target_dt = []

        for seq_time, seq_type, seq_dt_normed in zip(self.time_seq, self.event_seq, self.normed_time_delta_seq):
            self.history_times.append(seq_time[:-target_length])
            self.target_times.append(seq_time[-target_length:])
            self.history_types.append(seq_type[:-target_length])
            self.target_types.append(seq_type[-target_length:])
            self.history_dt.append(seq_dt_normed[:-target_length])
            self.target_dt.append(seq_dt_normed[-target_length:])

        self.seq_lengths = [seq.size(0) for seq in self.history_times]
        self.length = len(self.history_times)

    def __getitem__(self, key):
        return self.history_times[key], self.history_types[key], self.history_dt[key], \
            self.target_times[key], self.target_types[key], self.target_dt[key], \
            self.num_types, self.device, self.seq_lengths[key]

    def __len__(self):
        return self.length

    def load_dataset_hypro_format(self, dataset_dir, dict_name):
        with open(dataset_dir, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)
    
    def load_dataset_mooc(self, dataset_dir):
        with open(dataset_dir, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')

        timestamps = data["timestamps"]
        types = data["types"]
        intervals = data["intervals"]
        lengths = data["lengths"]
        t_max = data["t_max"]

        data_list = []  
        n_seq = len(lengths)

        for i in range(n_seq):
            L = int(lengths[i])
            if L <= 20:
                continue

            seq_t = np.asarray(timestamps[i]) * 10 / t_max
            seq_dt = np.asarray(intervals[i]) * 10 / t_max
            seq_y = np.asarray(types[i])
            

            if (seq_y >= 50).any():
                continue

            seq = []
            for t, dt, y in zip(seq_t.tolist(), seq_dt.tolist(), seq_y.tolist()):
                seq.append(
                    {
                        "time_since_start": float(t),
                        "time_since_last_event": float(dt),
                        "type_event": int(y),
                    }
                )

            data_list.append(seq)
        num_types = 50
        return data_list, num_types
    



class SeqDataset(Dataset):
    def __init__(self, dataset_dir, mode, t_scale=1, device=None, data_name='taxi'):
        assert mode in {'train', 'dev', 'test'}
        print(f'Loading {mode} dataset for {data_name}')
        
        if data_name == 'mooc':
            dataset_dir = os.path.join(dataset_dir, '{}.pkl'.format(mode))
            self.data, self.num_types = self.load_dataset_mooc(dataset_dir)
        else:
            dataset_dir = os.path.join(dataset_dir, '{}.pkl'.format(mode))
            self.data, self.num_types = self.load_dataset_hypro_format(dataset_dir, mode)

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # time step
        time_seq = [[x["time_since_start"] for x in seq] for seq in self.data]
        self.time_seq = [torch.tensor(seq[1:]) for seq in time_seq]

        # event type
        event_seq = [[x["type_event"] for x in seq] for seq in self.data]
        self.event_seq = [torch.tensor(seq[1:]) for seq in event_seq]

        # inter-arrival time
        time_delta_seq = [[x["time_since_last_event"] for x in seq] for seq in self.data]
        self.unnormed_time_delta_seq = [torch.tensor(seq[1:]) for seq in time_delta_seq]

        if mode == 'train':
            t_scale = np.max([sum(seq) for seq in self.unnormed_time_delta_seq])
            self.t_scale = t_scale
        else:
            self.t_scale = t_scale

        self.normed_time_delta_seq = [seq / self.t_scale for seq in self.unnormed_time_delta_seq]

        self.seq_lengths = [seq.size(0) for seq in self.time_seq]
        self.length = len(self.time_seq)

    def __getitem__(self, key):
        return self.time_seq[key], self.event_seq[key], self.normed_time_delta_seq[key], \
            self.num_types, self.device, self.seq_lengths[key]

    def __len__(self):
        return self.length

    def load_dataset_hypro_format(self, dataset_dir, dict_name):
        with open(dataset_dir, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    def load_dataset_mooc(self, dataset_dir):
        with open(dataset_dir, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')

        timestamps = data["timestamps"]
        types = data["types"]
        intervals = data["intervals"]
        lengths = data["lengths"]
        t_max = data["t_max"]

        data_list = []  
        n_seq = len(lengths)

        for i in range(n_seq):
            L = int(lengths[i])
            if L <= 20:
                continue

            seq_t = np.asarray(timestamps[i]) * 10 / t_max
            seq_dt = np.asarray(intervals[i]) * 10 / t_max
            seq_y = np.asarray(types[i])
            

            if (seq_y >= 50).any():
                continue

            seq = []
            for t, dt, y in zip(seq_t.tolist(), seq_dt.tolist(), seq_y.tolist()):
                seq.append(
                    {
                        "time_since_start": float(t),
                        "time_since_last_event": float(dt),
                        "type_event": int(y),
                    }
                )

            data_list.append(seq)
        num_types = 50
        return data_list, num_types


def load_dataset(dataset_dir, mode, target_length=20, t_scale=1, device=None, data_name=None, task='conditional'):
    if task == 'conditional':
        dataset = CondSeqDataset(
            dataset_dir=dataset_dir, mode=mode, target_length=target_length, t_scale=t_scale, device=device, data_name=data_name
        )
    elif task == 'unconditional':
        dataset = SeqDataset(
            dataset_dir=dataset_dir, mode=mode, t_scale=t_scale, device=device, data_name=data_name
        )
    else:
        raise ValueError(f'Unknown task: {task}')
    return dataset


class condBatch:
    def __init__(self, history_times, history_types, history_dt, target_times, target_types, target_dt, seq_lengths):
        self.history_times = history_times
        self.history_types = history_types.long()
        self.history_dt = history_dt
        self.target_times = target_times.long()
        self.target_types = target_types
        self.target_dt = target_dt
        self.seq_lengths = seq_lengths

class uncondBatch:
    def __init__(self, time_seq, event_seq, normed_time_delta_seq, seq_lengths, normed_final_time):
        self.time_seq = time_seq
        self.event_seq = event_seq.long()
        self.normed_time_delta_seq = normed_time_delta_seq
        self.seq_lengths = seq_lengths
        self.normed_final_time = normed_final_time


def create_collate_cond(block_size):
    def smart_pad_sequence(sequences, padding_value=0.0):
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        if block_size is not None:
            current_len = padded.size(1)

            remainder = current_len % block_size
            if remainder != 0:
                pad_len = block_size - remainder

                pad_shape = (padded.size(0), pad_len, *padded.shape[2:])
                padding = torch.full(pad_shape, padding_value,
                                     dtype=padded.dtype, device=padded.device)
                padded = torch.cat([padded, padding], dim=1)

        return padded

    def collate_cond(batch):
        num_types = batch[0][6]
        device = batch[0][7]

        history_times = [item[0] for item in batch]
        history_types = [item[1] for item in batch]
        history_dt = [item[2] for item in batch]

        target_times = [item[3] for item in batch]
        target_types = [item[4] for item in batch]
        target_dt = [item[5] for item in batch]

        seq_lengths = torch.tensor([item[8] for item in batch])

        history_times = smart_pad_sequence(history_times, padding_value=0.0)
        history_dt = smart_pad_sequence(history_dt, padding_value=0.0)
        history_types = smart_pad_sequence(history_types, padding_value=num_types)

        target_times = smart_pad_sequence(target_times, padding_value=0.0)
        target_dt = smart_pad_sequence(target_dt, padding_value=0.0)
        target_types = smart_pad_sequence(target_types, padding_value=num_types)


        return condBatch(
            history_times.to(device),
            history_types.to(device),
            history_dt.to(device),
            target_times.to(device),
            target_types.to(device),
            target_dt.to(device),
            seq_lengths.to(device)
        )

    return collate_cond


def create_collate_cond_test(block_size):
    def smart_pad_sequence(sequences, padding_value=0.0):
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        if block_size is not None:
            current_len = padded.size(1)

            remainder = current_len % block_size
            if remainder != 0:
                pad_len = block_size - remainder

                pad_shape = (padded.size(0), pad_len, *padded.shape[2:])
                padding = torch.full(pad_shape, padding_value,
                                     dtype=padded.dtype, device=padded.device)
                padded = torch.cat([padded, padding], dim=1)

        return padded
    
    def collate_cond_test(batch):
        num_types = batch[0][6]
        device = batch[0][7]

        history_times = [item[0] for item in batch]
        history_types = [item[1] for item in batch]
        history_dt = [item[2] for item in batch]

        target_times = [item[3] for item in batch]
        target_types = [item[4] for item in batch]
        target_dt = [item[5] for item in batch]

        seq_lengths = torch.tensor([item[8] for item in batch])

        history_times = smart_pad_sequence(history_times, padding_value=0.0)
        history_dt = smart_pad_sequence(history_dt, padding_value=0.0)
        history_types = smart_pad_sequence(history_types, padding_value=num_types)

        target_times = torch.stack(target_times).to(device)
        target_dt = torch.stack(target_dt).to(device)
        target_types = torch.stack(target_types).to(device)

        return condBatch(
            history_times.to(device),
            history_types.to(device),
            history_dt.to(device),
            target_times.to(device),
            target_types.to(device),
            target_dt.to(device),
            seq_lengths.to(device)
        )

    return collate_cond_test



def create_collate_uncond(block_size, args):
    def smart_pad_sequence(sequences, padding_value=0.0):
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        if block_size is not None:
            current_len = padded.size(1)

            remainder = current_len % block_size
            if remainder != 0:
                pad_len = block_size - remainder

                pad_shape = (padded.size(0), pad_len, *padded.shape[2:])
                padding = torch.full(pad_shape, padding_value,
                                     dtype=padded.dtype, device=padded.device)
                padded = torch.cat([padded, padding], dim=1)

        return padded

    def collate_uncond(batch):
        num_types = batch[0][3]
        device = batch[0][4]

        time_seq = [item[0] for item in batch]
        event_seq = [item[1] for item in batch]
        normed_time_delta_seq = [item[2] for item in batch]

        seq_lengths = torch.tensor([item[5] for item in batch])
        normed_final_time = torch.tensor([sum(seq) for seq in normed_time_delta_seq])

        time_seq = smart_pad_sequence(time_seq, padding_value=0.0)
        normed_time_delta_seq = smart_pad_sequence(normed_time_delta_seq, padding_value=0.0)
        event_seq = smart_pad_sequence(event_seq, padding_value=num_types)

        return uncondBatch(
            time_seq.to(device),
            event_seq.to(device),
            normed_time_delta_seq.to(device),
            seq_lengths.to(device),
            normed_final_time.to(device)
        )

    return collate_uncond