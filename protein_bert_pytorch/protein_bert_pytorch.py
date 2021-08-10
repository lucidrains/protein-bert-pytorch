import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

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

class GlobalLinearSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask = None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        out = einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head = 64,
        qk_activation = nn.Tanh()
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))

    def forward(self, x, context, mask = None, context_mask = None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        null_k, null_v = map(lambda t: repeat(t, 'd -> b h () d', b = b, h = h), (self.null_key, self.null_value))
        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype = torch.bool, device = device)

            if exists(context_mask):
                context_mask = F.pad(context_mask, (1, 0), value = True)
            else:
                context_mask = torch.ones(b, j, dtype = torch.bool, device = device)

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
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        glu_conv = False
    ):
        super().__init__()

        self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None

        conv_mult = 2 if glu_conv else 1

        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        self.local_to_global_attn = local_to_global_attn

        if local_to_global_attn:
            self.extract_global_info = CrossAttention(
                dim = dim,
                dim_keys = dim_global,
                dim_out = dim,
                heads = attn_heads,
                dim_head = attn_dim_head
            )
        else:
            self.extract_global_info = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
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

        self.global_attend_local = CrossAttention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head, qk_activation = attn_qk_activation)

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
            nn.LayerNorm(dim_global),
        )

    def forward(self, tokens, annotation, mask = None):
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation, mask = mask)
        else:
            global_info = self.extract_global_info(annotation)

        # process local (protein sequence)

        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0

        conv_input = rearrange(tokens, 'b n d -> b d n')

        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn
        tokens = self.local_norm(tokens)

        tokens = self.local_feedforward(tokens)

        # process global (annotations)

        annotation = self.global_attend_local(annotation, tokens, context_mask = mask)
        annotation = self.global_dense(annotation)
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation

# main model

class ProteinBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = 26,
        num_annotation = 8943,
        dim = 512,
        dim_global = 256,
        depth = 6,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        num_global_tokens = 1,
        glu_conv = False
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.num_global_tokens = num_global_tokens
        self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)

        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel, attn_qk_activation = attn_qk_activation, local_to_global_attn = local_to_global_attn, local_self_attn = local_self_attn, glu_conv = glu_conv) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens)

        self.to_annotation_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim_global, num_annotation)
        )

    def forward(self, seq, annotation, mask = None):
        tokens = self.token_emb(seq)

        annotation = self.to_global_emb(annotation)
        annotation = rearrange(annotation, 'b (n d) -> b n d', n = self.num_global_tokens)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)

        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        return tokens, annotation

# pretraining wrapper

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class PretrainingWrapper(nn.Module):
    def __init__(
        self,
        model,
        random_replace_token_prob = 0.05,
        remove_annotation_prob = 0.25,
        add_annotation_prob = 0.01,
        remove_all_annotations_prob = 0.5,
        seq_loss_weight = 1.,
        annotation_loss_weight = 1.,
        exclude_token_ids = (0, 1, 2)   # for excluding padding, start, and end tokens from being masked
    ):
        super().__init__()
        assert isinstance(model, ProteinBERT), 'model must be an instance of ProteinBERT'

        self.model = model

        self.random_replace_token_prob = random_replace_token_prob
        self.remove_annotation_prob = remove_annotation_prob
        self.add_annotation_prob = add_annotation_prob
        self.remove_all_annotations_prob = remove_all_annotations_prob

        self.seq_loss_weight = seq_loss_weight
        self.annotation_loss_weight = annotation_loss_weight

        self.exclude_token_ids = exclude_token_ids

    def forward(self, seq, annotation, mask = None):
        batch_size, device = seq.shape[0], seq.device

        seq_labels = seq
        annotation_labels = annotation

        if not exists(mask):
            mask = torch.ones_like(seq).bool()

        # prepare masks for noising sequence

        excluded_tokens_mask = mask

        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)

        random_replace_token_prob_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.random_replace_token_prob)

        # prepare masks for noising annotation

        batch_mask = torch.ones(batch_size, device = device, dtype = torch.bool)
        batch_mask = rearrange(batch_mask, 'b -> b ()')
        remove_annotation_from_batch_mask = get_mask_subset_with_prob(batch_mask, self.remove_all_annotations_prob)

        annotation_mask = annotation > 0
        remove_annotation_prob_mask = get_mask_subset_with_prob(annotation_mask, self.remove_annotation_prob)
        add_annotation_prob_mask = get_mask_subset_with_prob(~annotation_mask, self.add_annotation_prob)
        remove_annotation_mask = remove_annotation_from_batch_mask & remove_annotation_prob_mask

        # generate random tokens

        random_tokens = torch.randint(0, self.model.num_tokens, seq.shape, device=seq.device)

        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)

        # noise annotation

        noised_annotation = annotation + add_annotation_prob_mask.type(annotation.dtype)
        noised_annotation = noised_annotation * remove_annotation_mask.type(annotation.dtype)

        # denoise with model

        seq_logits, annotation_logits = self.model(noised_seq, noised_annotation, mask = mask)

        # calculate loss

        seq_logits = seq_logits[mask]
        seq_labels = seq_labels[mask]

        seq_loss = F.cross_entropy(seq_logits, seq_labels, reduction = 'sum')
        annotation_loss = F.binary_cross_entropy_with_logits(annotation_logits, annotation_labels, reduction = 'sum')

        return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight
