import math
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_blocks(a, t, x_shape):
    batch, block_nums = t.shape
    _, seq_len, _ = x_shape
    assert seq_len % block_nums == 0
    block_size = seq_len // block_nums
    out = a.gather(-1, t.flatten()).reshape(batch, block_nums)
    out = out.repeat_interleave(block_size, dim=1)
    return out.unsqueeze(-1)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ContinuousSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_coeff = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -emb_coeff)
        emb = t.unsqueeze(-1) * frequencies
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionModel(Module):
    def __init__(
            self,
            model,
            block_size=4,
            num_classes=5,
            transformer_dim=32,
            timesteps=100,
            sampling_timesteps=None,
            device='cuda',
            objective='pred_x0',
            beta_schedule='sigmoid',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            min_snr_loss_weight=False,
            min_snr_gamma=5,
            t_emb_lambda=0
    ):
        super().__init__()

        self.model = model
        self.block_size = block_size
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.device = device
        self.objective = objective
        self.t_emb_lambda = t_emb_lambda

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # time embedding
        self.time_pos_emb = ContinuousSinusoidalPosEmb(transformer_dim)
        self.mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.Softplus(),
            nn.Linear(transformer_dim, transformer_dim)
        )

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract_blocks(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract_blocks(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        t_emb = self.time_pos_emb(t)
        t_emb = self.mlp(t_emb).unsqueeze(1)
        x = x + self.t_emb_lambda * t_emb

        model_output = self.model(x, sample_mode=True)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            raise ValueError(f'Objective {self.objective} not supported')

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t: int):
        b = x.shape[0]
        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_seqs = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_seqs

    @torch.inference_mode()
    def cond_p_sample_loop(self, batch_size, tgt_len, dim):
        tgt_num_blocks = (tgt_len + self.block_size - 1) // self.block_size
        tgt_seqs = None
        for i in range(tgt_num_blocks):
            block = torch.randn((batch_size, self.block_size, dim), device=self.device)

            for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                block = self.p_sample(block, t)

            self.model(block, sample_mode=True, store_kv=True)

            if tgt_seqs is None:
                tgt_seqs = block
            else:
                tgt_seqs = torch.cat((tgt_seqs, block), dim=1)

        return tgt_seqs[:, :tgt_len, :].clone()

    @torch.inference_mode()
    def cond_ddim_sample(self, batch_size, tgt_len, dim):
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        tgt_num_blocks = (tgt_len + self.block_size - 1) // self.block_size
        tgt_seqs = None

        for i in range(tgt_num_blocks):
            block = torch.randn((batch_size, self.block_size, dim), device=self.device)

            for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
                time_cond = torch.full((batch_size,), time, device=self.device)
                pred_noise, x_start = self.model_predictions(block, time_cond)

                if time_next < 0:
                    block = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(block)

                block = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            self.model(block, sample_mode=True, store_kv=True)

            if tgt_seqs is None:
                tgt_seqs = block
            else:
                tgt_seqs = torch.cat((tgt_seqs, block), dim=1)

        return tgt_seqs[:, :tgt_len, :].clone()

    @torch.inference_mode()
    def cond_sample(self, seq_hidden, tgt_len, batch_size=32):
        b, s, d = seq_hidden.shape
        assert b == batch_size and s % self.block_size == 0
        self.model.batch_size = batch_size
        self.model.reset_kv_cache()

        num_blocks = s // self.block_size

        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = (i + 1) * self.block_size
            cond_block = seq_hidden[:, start_idx:end_idx, :]
            self.model(cond_block, sample_mode=True, store_kv=True)

        sample_fn = self.cond_p_sample_loop if not self.is_ddim_sampling else self.cond_ddim_sample
        return sample_fn(b, tgt_len, d)

    @torch.inference_mode()
    def uncond_p_sample_loop(self, batch_size, dim):
        block = torch.randn((batch_size, self.block_size, dim), device=self.device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            block = self.p_sample(block, t)

        self.model(block, sample_mode=True, store_kv=True)


        return block

    @torch.inference_mode()
    def uncond_ddim_sample(self, batch_size, dim):
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        block = torch.randn((batch_size, self.block_size, dim), device=self.device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch_size,), time, device=self.device)
            pred_noise, x_start = self.model_predictions(block, time_cond)

            if time_next < 0:
                block = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(block)

            block = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        self.model(block, sample_mode=True, store_kv=True)

        return block

    @torch.inference_mode()
    def uncond_sample(self, batch_size=32):
        sample_fn = self.uncond_p_sample_loop if not self.is_ddim_sampling else self.uncond_ddim_sample
        return sample_fn(batch_size, self.transformer_dim)

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract_blocks(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_blocks(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def keep_last_tgt_len(self, mask, tgt_len):
        batch_size, seq_len = mask.shape

        if tgt_len >= seq_len:
            return mask.clone()

        cumsum = mask.cumsum(dim=1)
        total_true = mask.sum(dim=1, keepdim=True)
        start_pos = total_true - int(tgt_len)
        start_pos = start_pos.clamp(min=0)

        new_mask = mask & (cumsum > start_pos)

        return new_mask

    def p_losses(self, x_start, t, pad_mask_seq, tgt_len, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t0 = t.clone()

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        b_t, block_nums, d_t = t.shape
        b_x, seq_len, d_x = x.shape
        assert b_t == b_x and d_t == d_x and seq_len % block_nums == 0
        block_size = seq_len // block_nums

        t_expanded = t.repeat_interleave(block_size, dim=1)
        x = x + self.t_emb_lambda * t_expanded

        mask = pad_mask_seq.unsqueeze(-1)
        x = x.masked_fill(mask, 0.0)

        assert x.shape == x_start.shape
        x_input = torch.cat([x, x_start], dim=1)
        model_out = self.model(x_input)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t0, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')

        non_padding_mask = ~pad_mask_seq

        if tgt_len > 0:
            non_padding_mask = self.keep_last_tgt_len(non_padding_mask, tgt_len)

        mask_expanded = non_padding_mask.unsqueeze(-1)
        mask_expanded = mask_expanded.expand(-1, -1, d_x)
        loss_masked = loss * mask_expanded.float()
        non_padding_count = non_padding_mask.float().sum()
        assert non_padding_count > 0

        return loss_masked.sum() / non_padding_count

    def forward(self, seq_hidden, pad_mask_seq, tgt_len=0):
        self.model.remove_kv_cache()
        b, s, d = seq_hidden.shape
        assert s % self.block_size == 0, f"seq_len {s} must be divisible by block_size {self.block_size}"
        block_num = s // self.block_size
        t = torch.randint(0, self.num_timesteps, (b, block_num), device=self.device).long()

        return self.p_losses(seq_hidden, t, pad_mask_seq, tgt_len)