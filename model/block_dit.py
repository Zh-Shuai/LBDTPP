import math
import typing

import einops
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    FLEX_ATTN_AVAILABLE = True
except:
    FLEX_ATTN_AVAILABLE = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.
        n: Sequence length of x_0 and x_t

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
            (block_q > block_kv)
            & (x0_flag_kv == 1)
            & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


# @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


def residual_linear(x, W, x_skip, residual_scale):
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(self, n, block_size, dim, n_heads, mlp_ratio=4, dropout=0.1, attn_backend='flex'):
        super().__init__()
        self.n = n
        self.block_size = block_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_backend = attn_backend
        self.kv_cache = None
        self.cache_idx = 0

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def get_qkv(self, x, store_kv=False):
        if self.kv_cache is not None:
            new_qkv = self.attn_qkv(x)
            self.kv_cache[:, self.cache_idx:self.cache_idx + self.block_size] = new_qkv
            qkv = self.kv_cache[:, :self.cache_idx + self.block_size].clone()
        else:
            qkv = self.attn_qkv(x)

        if store_kv:
            self.cache_idx += self.block_size
            if self.cache_idx >= self.n:
                self.cache_idx = self.n - self.block_size
                self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)

        return qkv

    def cross_attn(self, qkv, mask=None):
        scale = qkv.shape[-1]
        qkv = qkv.transpose(1, 3)
        mask = mask.bool().to(qkv.device) if mask is not None else None
        x = F.scaled_dot_product_attention(
            query=qkv[:, :, 0],
            key=qkv[:, :, 1],
            value=qkv[:, :, 2],
            attn_mask=mask,
            is_causal=False,
            scale=1 / math.sqrt(scale))
        x = x.transpose(1, 2)
        x = einops.rearrange(x, 'b s h d -> b s (h d)')
        return x

    def cross_attn_flex(self, qkv, mask=None):
        qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
        mask = mask.to(qkv.device) if mask is not None else None
        x = fused_flex_attention(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
        x = einops.rearrange(x, 'b h s d -> b s (h d)')
        return x

    def forward(self, x, mask=None, sample_mode=False, store_kv=False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if mask is not None and not sample_mode:
            n = mask.shape[-1] // 2
            qkv_x = self.get_qkv(x[:, :n])
            qkv_x0 = self.get_qkv(x[:, n:])
            qkv = torch.cat((qkv_x, qkv_x0), dim=1)
        else:
            qkv = self.get_qkv(x, store_kv=store_kv)

        if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            x = self.cross_attn_flex(qkv, mask=mask)
        elif self.attn_backend == 'sdpa' or not FLEX_ATTN_AVAILABLE:
            x = self.cross_attn(qkv, mask=mask)
        else:
            raise ValueError('Unknown attention backend')

        if self.kv_cache is not None:
            x = x[:, -self.block_size:]

        x = bias_dropout_scale_fn(self.attn_out(x), None, torch.ones_like(x), x_skip, self.dropout)
        x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, torch.ones_like(x), x, self.dropout)

        return x


class BlockDIT(nn.Module):
    def __init__(self, model_length=128, block_size=4, transformer_dim=32, transformer_heads=2, n_decoder_layers=1,
                 dropout=0.1, batch_size=32, device='cuda'):
        super().__init__()
        self.n = model_length
        self.block_size = block_size
        self.hidden_dim = transformer_dim
        self.n_heads = transformer_heads
        self.sigma_map = TimestepEmbedder(transformer_dim)
        self.dropout = dropout
        self.batch_size = batch_size
        self.attn_backend = 'sdpa'
        self.device = device

        layers = []
        for _ in range(n_decoder_layers):
            layer = DDiTBlock(self.n,
                              self.block_size,
                              self.hidden_dim,
                              self.n_heads,
                              dropout=self.dropout,
                              attn_backend=self.attn_backend)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def gen_mask(self, seqlen, block_size, attn_backend='flex'):
        if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            self.block_diff_mask = create_block_mask(
                partial(block_diff_mask, block_size=block_size, n=seqlen),
                B=None, H=None, Q_LEN=seqlen * 2, KV_LEN=seqlen * 2)
        elif attn_backend == 'sdpa' or not FLEX_ATTN_AVAILABLE:
            self.block_diff_mask = block_diff_mask(
                b=None, h=None, q_idx=torch.arange(seqlen * 2)[:, None],
                kv_idx=torch.arange(seqlen * 2)[None, :], block_size=block_size, n=seqlen)
        else:
            raise ValueError('Unknown attention backend')

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.kv_cache = torch.zeros(
                self.batch_size,
                self.n,
                self.hidden_dim * 3,
                device=self.device,
                dtype=torch.bfloat16)
            layer.cache_idx = 0

    def remove_kv_cache(self):
        for layer in self.layers:
            layer.kv_cache = None
            layer.cache_idx = 0

    def forward(self, x, sample_mode=False, store_kv=False):
        seq_len = self.n
        if sample_mode:
            mask = None
        else:
            assert x.shape[1] % 2 == 0
            seq_len = x.shape[1] // 2
            self.gen_mask(seq_len, self.block_size, attn_backend=self.attn_backend)
            mask = self.block_diff_mask

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.layers)):
                x = self.layers[i](
                    x,
                    sample_mode=sample_mode,
                    mask=mask,
                    store_kv=store_kv)

        if not sample_mode:
            x = x[:, :seq_len]
        return x