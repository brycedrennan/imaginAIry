import math
from functools import lru_cache

import psutil
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from imaginairy.modules.diffusion.util import checkpoint as checkpoint_eval
from imaginairy.utils import get_device

XFORMERS_IS_AVAILABLE = False

try:
    if get_device() == "cuda":
        import xformers
        import xformers.ops

        XFORMERS_IS_AVAILABLE = True
except ImportError:
    pass


ALLOW_SPLITMEM = True
ATTENTION_PRECISION_OVERRIDE = "default"


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


def get_mem_free_total(device):
    device_type = "mps" if device.type == "mps" else "cuda"
    if device_type == "cuda":
        stats = torch.cuda.memory_stats(device)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        mem_free_total *= 0.9
    else:
        # if we don't add a buffer, larger images come out as noise
        mem_free_total = psutil.virtual_memory().available * 0.6

    return mem_free_total


@lru_cache(maxsize=1)
def get_mps_gb_ram():
    return psutil.virtual_memory().total / (1024**3)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        # from imaginairy.api import _global_mask_hack
        #
        # if mask is None and _global_mask_hack is not None:
        #     mask = _global_mask_hack.to(torch.bool)

        if get_device() == "cuda" or "mps" in get_device():  # noqa
            if not XFORMERS_IS_AVAILABLE and ALLOW_SPLITMEM:
                return self.forward_splitmem(x, context=context, mask=mask)

        h = self.heads
        # print(x.shape)

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context) * self.scale
        v = self.to_v(context)

        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        # force cast to fp32 to avoid overflowing
        if ATTENTION_PRECISION_OVERRIDE == "fp32":
            with torch.autocast(enabled=False, device_type=get_device()):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k)
        else:
            sim = einsum("b i d, b j d -> b i j", q, k)
        # print(sim.shape)
        # print("*" * 100)
        del q, k
        # if mask is not None:
        #     if sim.shape[2] == 320 and False:
        #         mask = [mask] * 2
        #         mask = rearrange(mask, "b ... -> b (...)")
        #         _max_neg_value = -torch.finfo(sim.dtype).max
        #         mask = repeat(mask, "b j -> (b h) () j", h=h)
        #         sim.masked_fill_(~mask, _max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

    def forward_splitmem(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = context if context is not None else x
        k_in = self.to_k(context) * self.scale
        v_in = self.to_v(context)
        del context, x

        q, k, v = (
            rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q_in, k_in, v_in)
        )
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

        mem_free_total = get_mem_free_total(q.device)

        gb = 1024**3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier

        steps = 1

        if "mps" in get_device():
            # https://github.com/brycedrennan/imaginAIry/issues/175
            # https://github.com/invoke-ai/InvokeAI/issues/1244
            mps_gb = get_mps_gb_ram()
            factor = 32 / mps_gb

            slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1] * 16 * factor))
        else:
            if mem_required > mem_free_total:
                steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

            if steps > 64:
                max_res = (
                    math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
                )
                msg = f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free"
                raise RuntimeError(msg)
            slice_size = (
                q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            )

        # steps = len(range(0, q.shape[1], slice_size))
        # print(f"Splitting attention into {steps} steps of {slice_size} slices")

        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            # force cast to fp32 to avoid overflowing
            if ATTENTION_PRECISION_OVERRIDE == "fp32":
                with torch.autocast(enabled=False, device_type=get_device()):
                    q, k = q.float(), k.float()
                    s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k)
            else:
                s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum("b i j, b j d -> b i d", s2, v)
            del s2

        del q, k, v

        r2 = rearrange(r1, "(b h) n d -> b n (h d)", h=h)
        del r1

        return self.to_out(r2)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # print(
        #     f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #     f"{heads} heads."
        # )
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = (
            t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous()
            for t in (q, k, v)
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if mask is not None:
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILABLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint_eval(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = x.contiguous() if x.device.type == "mps" else x
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs.
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if self.use_linear:
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            x = self.proj_in(x)
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, context=context[i])
            x = self.proj_out(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        else:
            x = self.proj_in(x)
            x = rearrange(x, "b c h w -> b (h w) c")
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, context=context[i])
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.proj_out(x)

        return x + x_in
