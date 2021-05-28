import torch
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head = 64
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_out)

    def forward(self, x, context, mask = None, context_mask = None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            if not exists(mask):
                mask = torch.ones(b, q.shape[-2], dtype = torch.bool, device = device)

            if not exists(context_mask):
                context_mask = torch.ones(b, k.shape[-2], dtype = torch.bool, device = device)

            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Layer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_global,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64
    ):
        super().__init__()
        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU()
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU()
        )

        self.global_to_local = nn.Sequential(
            nn.Linear(dim_global, dim),
            nn.GELU(),
            Rearrange('b d -> b () d')
        )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )),
            nn.LayerNorm(dim)
        )

        self.global_attend_local = Attention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head)

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU()
        )

        self.global_norm = nn.LayerNorm(dim_global)

        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim_global, dim_global),
                nn.GELU()
            )),
            nn.LayerNorm(dim_global)
        )

    def forward(self, tokens, annotation, mask = None):

        global_info = self.global_to_local(annotation)

        # process local (protein sequence)

        conv_input = rearrange(tokens, 'b n d -> b d n')
        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out + global_info
        tokens = self.local_norm(tokens)

        tokens = self.local_feedforward(tokens)

        # process global (annotations)

        one_global_token = rearrange(annotation, 'b d -> b () d')

        local_info = self.global_attend_local(one_global_token, tokens, context_mask = mask)
        annotation = self.global_dense(annotation)

        annotation = annotation + rearrange(local_info, 'b () d -> b d')
        annotation = self.global_norm(annotation)

        annotation = self.global_feedforward(annotation)
        return tokens, annotation

class ProteinBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = 21,
        num_annotation = 8943,
        dim = 512,
        dim_global = 256,
        depth = 6,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_global_emb = nn.Linear(num_annotation, dim_global)

        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens)
        self.to_annotation_logits = nn.Linear(dim_global, num_annotation)

    def forward(self, seq, annotation, mask = None):
        tokens = self.token_emb(seq)
        annotation = self.to_global_emb(annotation)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)

        return self.to_token_logits(tokens), self.to_annotation_logits(annotation)
