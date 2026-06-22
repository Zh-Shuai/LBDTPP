import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block_dit import BlockDIT
from model.diffusion_model import DiffusionModel


class BDTPPModel(nn.Module):
    def __init__(self, args, num_classes):
        super(BDTPPModel, self).__init__()
        assert num_classes == args.num_classes

        self.loss_lambda = args.loss_lambda
        model_length = args.model_length
        block_size = args.block_size
        transformer_dim = args.transformer_dim
        transformer_heads = args.transformer_heads
        num_decoder_layers = args.num_decoder_layers
        dropout = args.dropout
        device = args.device
        num_timesteps = args.diffusion_steps
        sampling_steps = args.sampling_steps
        batch_size = args.batch_size
        self.num_classes = num_classes
        self.tgt_len = args.tgt_len
        self.device = device

        self.block_dit = BlockDIT(model_length=model_length, block_size=block_size, transformer_dim=transformer_dim,
                                  transformer_heads=transformer_heads, n_decoder_layers=num_decoder_layers,
                                  dropout=dropout, batch_size=batch_size, device=device)

        self.diffusion_model = DiffusionModel(self.block_dit, block_size=block_size, num_classes=num_classes,
                                              transformer_dim=transformer_dim,
                                              timesteps=num_timesteps, sampling_timesteps=sampling_steps, device=device)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / int(transformer_dim)) for i in range((int(transformer_dim)))],
            device=torch.device(self.device))


        self.event_emb = nn.Embedding(num_classes + 1, int(transformer_dim), padding_idx=num_classes)
        self.event_emb.weight.requires_grad = False


        self.time_decoder = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim, 1),
            nn.Softplus()
        )

        self.type_decoder = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim, num_classes)
        )


    def _init_weights(self):
        for module in [self.type_decoder, self.time_decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)


    def temporal_enc(self, dt_seq):
        result = dt_seq.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def get_pad_mask(self, seq):
        assert seq.dim() == 2
        return seq.eq(self.num_classes)

    def find_first_exceed_index(self, cumulative_x, stop_time):
        b, s = cumulative_x.shape
        stop_time_expanded = stop_time.view(-1, 1).expand(-1, s)

        mask = cumulative_x > stop_time_expanded
        first_true_idx = torch.argmax(mask.int(), dim=1)

        no_exceed_mask = ~mask.any(dim=1)
        first_true_idx[no_exceed_mask] = s - 1

        exceeded_time_index = first_true_idx - 1
        exceeded_time_index = torch.clamp(exceeded_time_index, min=0)

        return exceeded_time_index

    def sequence_truncation(self, seqs, exceeded_time_index):
        b = seqs.shape[0]
        new_seqs = []

        for i in range(b):
            idx = exceeded_time_index[i].item()
            trunc_seq = seqs[i, :idx + 1]
            new_seqs.append(trunc_seq)

        return new_seqs

    def forward(self, x, e, tgt_x=None, tgt_e=None, task='conditional'):
        if task == 'unconditional':
            # Mask
            pad_mask_seq = self.get_pad_mask(e)

            # Encoding
            seq_hidden = self.temporal_enc(x) + self.event_emb(e)
            mask = pad_mask_seq.unsqueeze(-1)
            seq_hidden = seq_hidden.masked_fill(mask, 0.0)

            # Diffusion
            diff_loss = self.diffusion_model(seq_hidden, pad_mask_seq)

            # Decoding
            time_pred = self.time_decoder(seq_hidden).squeeze(-1)
            type_logits = self.type_decoder(seq_hidden)

            non_padding_mask = ~pad_mask_seq
            non_padding_count = non_padding_mask.float().sum()
            assert non_padding_count > 0

            time_loss_elementwise = F.mse_loss(time_pred, x, reduction='none')
            time_loss = (time_loss_elementwise * non_padding_mask.float()).sum() / non_padding_count

            type_loss = F.cross_entropy(
                type_logits.reshape(-1, type_logits.size(-1)),
                e.reshape(-1),
                ignore_index=self.num_classes,
                reduction='mean'
            )

            loss = diff_loss + (time_loss + type_loss) * self.loss_lambda

        elif task == 'conditional':
            if tgt_x is None or tgt_e is None:
                raise ValueError(
                    "Conditional generation task requires 'tgt_x', 'tgt_e', and 'hist_time_stamps' parameters.")

            # Mask
            x_seq = torch.cat([x, tgt_x], dim=1)
            e_seq = torch.cat([e, tgt_e], dim=1)
            pad_mask_seq = self.get_pad_mask(e_seq)

            # Encoding
            seq_hidden = self.temporal_enc(x_seq) + self.event_emb(e_seq)
            mask = pad_mask_seq.unsqueeze(-1)
            seq_hidden = seq_hidden.masked_fill(mask, 0.0)

            # Diffusion
            diff_loss = self.diffusion_model(seq_hidden, pad_mask_seq, tgt_len=self.tgt_len)

            # Decoding
            time_pred = self.time_decoder(seq_hidden).squeeze(-1)
            type_logits = self.type_decoder(seq_hidden)

            non_padding_mask = ~pad_mask_seq
            non_padding_count = non_padding_mask.float().sum()
            assert non_padding_count > 0

            time_loss_elementwise = F.mse_loss(time_pred, x_seq, reduction='none')
            time_loss = (time_loss_elementwise * non_padding_mask.float()).sum() / non_padding_count

            type_loss = F.cross_entropy(
                type_logits.reshape(-1, type_logits.size(-1)),
                e_seq.reshape(-1),
                ignore_index=self.num_classes,
                reduction='mean'
            )

            loss = diff_loss + (time_loss + type_loss) * self.loss_lambda
        else:
            raise ValueError(f"Unsupported generation task: '{task}'. Please use 'conditional' or 'unconditional'.")

        return loss, diff_loss, time_loss, type_loss

    def conditional_generation_sample(self, hist_x, hist_e, tgt_len, batch_size):
        pad_mask_seq = self.get_pad_mask(hist_e)
        seq_hidden = self.temporal_enc(hist_x) + self.event_emb(hist_e)
        mask = pad_mask_seq.unsqueeze(-1)
        seq_hidden = seq_hidden.masked_fill(mask, 0.0)
        seq_sample = self.diffusion_model.cond_sample(seq_hidden, tgt_len, batch_size=batch_size)
        pred_x = self.time_decoder(seq_sample).squeeze(-1)
        type_logits = self.type_decoder(seq_sample)
        pred_e = type_logits.argmax(dim=-1)

        return pred_x, pred_e

    def unconditional_generation_sample(self, stop_time, batch_size):
        assert stop_time.shape[0] == batch_size
        self.block_dit.batch_size = batch_size
        self.block_dit.reset_kv_cache()
        pred_x, pred_e = None, None
        accumulated_time = torch.zeros_like(stop_time, device=self.device)

        i = 0

        while not (accumulated_time > stop_time).all().item() and i < 500:
            seq_sample = self.diffusion_model.uncond_sample(batch_size=batch_size)
            gen_x = self.time_decoder(seq_sample).squeeze(-1)
            type_logits = self.type_decoder(seq_sample)
            gen_e = type_logits.argmax(dim=-1)

            if pred_x is None:
                pred_x = gen_x
            else:
                pred_x = torch.cat((pred_x, gen_x), dim=1)

            if pred_e is None:
                pred_e = gen_e
            else:
                pred_e = torch.cat((pred_e, gen_e), dim=1)

            accumulated_time = accumulated_time + gen_x.sum(dim=1)
            remaining_time = stop_time - accumulated_time
            max_remaining = remaining_time.max().item()
            print(f"max stop_time: {max(stop_time)}")
            print(f"Iteration {i}: max time difference = {max_remaining}")

            i += 1

        cumulative_x = torch.cumsum(pred_x, dim=1)
        exceeded_time_index = self.find_first_exceed_index(cumulative_x, stop_time)

        pred_x = self.sequence_truncation(pred_x, exceeded_time_index)
        pred_e = self.sequence_truncation(pred_e, exceeded_time_index)

        return pred_x, pred_e


def get_model(args, num_classes):
    return BDTPPModel(args, num_classes)


def get_model_id(args):
    return 'block_diffusion_tpp_{}_tgt_len_{}'.format(args.diffusion_steps, args.tgt_len)
