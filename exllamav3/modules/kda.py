from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from ..model.model_tp_alloc import TPAllocation
from .rmsnorm import RMSNorm
from ..cache import CacheableState

"""
Kimi Delta Attention (KDA) module for Kimi architecture.

This is a linear attention mechanism using FLA's KDA operations.
It uses separate convolutions for Q/K/V and a different gate mechanism
than GatedDeltaNet.
"""

# Import FLA operations
try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    chunk_kda = None
    fused_recurrent_kda = None
    fused_kda_gate = None
    ShortConvolution = None
    FusedRMSNormGated = None


class KDA_RecurrentState(CacheableState):
    """Recurrent state for KDA attention."""

    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        conv_state_q: torch.Tensor = None,
        conv_state_k: torch.Tensor = None,
        conv_state_v: torch.Tensor = None,
        recurrent_state: torch.Tensor = None,
        batched=False
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.conv_state_q = conv_state_q
        self.conv_state_k = conv_state_k
        self.conv_state_v = conv_state_v
        self.recurrent_state = recurrent_state
        self.batched = batched

    @override
    def stash(self):
        return KDA_RecurrentState(
            self.position,
            self.positions,
            self.conv_state_q.cpu() if self.conv_state_q is not None else None,
            self.conv_state_k.cpu() if self.conv_state_k is not None else None,
            self.conv_state_v.cpu() if self.conv_state_v is not None else None,
            self.recurrent_state.cpu() if self.recurrent_state is not None else None,
        )

    @override
    def unstash(self, device):
        return KDA_RecurrentState(
            self.position,
            self.positions,
            self.conv_state_q.to(device, non_blocking=True) if self.conv_state_q is not None else None,
            self.conv_state_k.to(device, non_blocking=True) if self.conv_state_k is not None else None,
            self.conv_state_v.to(device, non_blocking=True) if self.conv_state_v is not None else None,
            self.recurrent_state.to(device, non_blocking=True) if self.recurrent_state is not None else None,
        )

    @override
    def get_size(self):
        size = 0
        for state in [self.conv_state_q, self.conv_state_k, self.conv_state_v, self.recurrent_state]:
            if state is not None:
                size += state.element_size() * state.numel()
        return size

    def collect_batch(self, batch: list[KDA_RecurrentState]):
        csq = torch.cat([b.conv_state_q for b in batch], dim=0) if batch[0].conv_state_q is not None else None
        csk = torch.cat([b.conv_state_k for b in batch], dim=0) if batch[0].conv_state_k is not None else None
        csv = torch.cat([b.conv_state_v for b in batch], dim=0) if batch[0].conv_state_v is not None else None
        rs = torch.cat([b.recurrent_state for b in batch], dim=0) if batch[0].recurrent_state is not None else None
        positions = [b.position for b in batch]
        return KDA_RecurrentState(None, positions, csq, csk, csv, rs, True)

    def distribute_batch(self, batch: list[KDA_RecurrentState]):
        for i, b in enumerate(batch):
            if self.conv_state_q is not None:
                b.conv_state_q.copy_(self.conv_state_q[i:i+1, ...])
            if self.conv_state_k is not None:
                b.conv_state_k.copy_(self.conv_state_k[i:i+1, ...])
            if self.conv_state_v is not None:
                b.conv_state_v.copy_(self.conv_state_v[i:i+1, ...])
            if self.recurrent_state is not None:
                b.recurrent_state.copy_(self.recurrent_state[i:i+1, ...])
            b.position = self.positions[i]


def prepare_for_kda_recurrence(input_ids: torch.Tensor, params: dict, model) -> torch.Tensor:
    """
    Add KDA recurrent state parameters to state.
    """
    batch_shape = params.get("batch_shape")
    cache_seqlens = params.get("cache_seqlens")

    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            rs = params.get("kda_recurrent_states")
            if rs is None:
                raise ValueError("Past length given, but no previous state for KDA in params")
            for k, v in rs.items():
                if not v.batched and v.position != past_len:
                    raise ValueError("KDA recurrent states don't match input past_len")
        else:
            rl = model.get_kda_layers()
            rs = {attn.layer_idx: KDA_RecurrentState() for attn in rl}
            params["kda_recurrent_states"] = rs

    elif cache_seqlens is not None:
        pass

    else:
        if "kda_recurrent_states" in params:
            raise ValueError("kda_recurrent_states given without bsz and seqlens")


class KDA(Module):
    """
    Kimi Delta Attention module using FLA's KDA operations.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        key_q: str | None = None,
        key_k: str | None = None,
        key_v: str | None = None,
        key_f_a: str | None = None,
        key_f_b: str | None = None,
        key_g_a: str | None = None,
        key_g_b: str | None = None,
        key_b: str | None = None,
        key_o: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "KDA"

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype

        self.projection_size = head_dim * num_heads

        # Q/K/V projections
        self.q_proj = Linear(
            config, f"{key}.{key_q}",
            hidden_size, self.projection_size,
            qmap=qmap + ".input" if qmap else None,
            out_dtype=torch.float
        )
        self.register_submodule(self.q_proj)

        self.k_proj = Linear(
            config, f"{key}.{key_k}",
            hidden_size, self.projection_size,
            qmap=qmap + ".input" if qmap else None,
            out_dtype=torch.float
        )
        self.register_submodule(self.k_proj)

        self.v_proj = Linear(
            config, f"{key}.{key_v}",
            hidden_size, self.projection_size,
            qmap=qmap + ".input" if qmap else None,
            out_dtype=torch.float
        )
        self.register_submodule(self.v_proj)

        # Gate projections (for decay gate g)
        self.f_a_proj = Linear(
            config, f"{key}.{key_f_a}",
            hidden_size, head_dim,
            qmap=None,
            out_dtype=torch.float
        )
        self.register_submodule(self.f_a_proj)

        self.f_b_proj = Linear(
            config, f"{key}.{key_f_b}",
            head_dim, self.projection_size,
            qmap=None,
            out_dtype=torch.float
        )
        self.register_submodule(self.f_b_proj)

        # Output gate projections
        self.g_a_proj = Linear(
            config, f"{key}.{key_g_a}",
            hidden_size, head_dim,
            qmap=None,
            out_dtype=torch.float
        )
        self.register_submodule(self.g_a_proj)

        self.g_b_proj = Linear(
            config, f"{key}.{key_g_b}",
            head_dim, self.projection_size,
            qmap=None,
            out_dtype=torch.float
        )
        self.register_submodule(self.g_b_proj)

        # Beta projection
        self.b_proj = Linear(
            config, f"{key}.{key_b}",
            hidden_size, num_heads,
            qmap=None,
            out_dtype=torch.float
        )
        self.register_submodule(self.b_proj)

        # Output projection
        self.o_proj = Linear(
            config, f"{key}.{key_o}",
            self.projection_size, hidden_size,
            qmap=qmap + ".output" if qmap else None,
            out_dtype=out_dtype
        )
        self.register_submodule(self.o_proj)

        # Tensors loaded separately
        self.A_log = None
        self.dt_bias = None
        self.key_A_log = f"{key}.A_log"
        self.key_dt_bias = f"{key}.dt_bias"

        # Short convolutions (created at load time using FLA)
        self.q_conv1d = None
        self.k_conv1d = None
        self.v_conv1d = None
        self.q_conv1d_weight = None
        self.k_conv1d_weight = None
        self.v_conv1d_weight = None

        # Output norm
        self.o_norm = RMSNorm(
            config, f"{key}.o_norm",
            rms_norm_eps=rms_norm_eps
        )
        self.register_submodule(self.o_norm)

        self.caps.update({
            "recurrent_cache": True
        })


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)

        # Load A_log and dt_bias
        self.A_log = self.config.stc.get_tensor(
            self.key_A_log, self.device, optional=False, allow_bf16=True
        )
        self.dt_bias = self.config.stc.get_tensor(
            self.key_dt_bias, self.device, optional=False, allow_bf16=True
        )

        # Load conv weights
        self.q_conv1d_weight = self.config.stc.get_tensor(
            f"{self.key}.q_conv1d.weight", self.device, optional=False, allow_bf16=True
        )
        self.k_conv1d_weight = self.config.stc.get_tensor(
            f"{self.key}.k_conv1d.weight", self.device, optional=False, allow_bf16=True
        )
        self.v_conv1d_weight = self.config.stc.get_tensor(
            f"{self.key}.v_conv1d.weight", self.device, optional=False, allow_bf16=True
        )

        # Create ShortConvolution modules with loaded weights
        if ShortConvolution is not None:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.projection_size,
                kernel_size=self.conv_kernel_size,
                activation='silu',
            ).to(device)
            self.q_conv1d.weight.data.copy_(self.q_conv1d_weight)

            self.k_conv1d = ShortConvolution(
                hidden_size=self.projection_size,
                kernel_size=self.conv_kernel_size,
                activation='silu',
            ).to(device)
            self.k_conv1d.weight.data.copy_(self.k_conv1d_weight)

            self.v_conv1d = ShortConvolution(
                hidden_size=self.projection_size,
                kernel_size=self.conv_kernel_size,
                activation='silu',
            ).to(device)
            self.v_conv1d.weight.data.copy_(self.v_conv1d_weight)

        self.o_norm.load(device, **kwargs)


    @override
    def unload(self):
        self.A_log = None
        self.dt_bias = None
        self.q_conv1d = None
        self.k_conv1d = None
        self.v_conv1d = None
        self.q_conv1d_weight = None
        self.k_conv1d_weight = None
        self.v_conv1d_weight = None
        self.o_norm.unload()
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if chunk_kda is None:
            raise RuntimeError("KDA requires fla-core package. Please install with: pip install -U fla-core")

        bsz, seqlen, _ = x.shape

        # Get recurrent state
        rs = params.get("kda_recurrent_states")
        if rs is not None:
            rs = rs[self.layer_idx]
            conv_state_q = rs.conv_state_q
            conv_state_k = rs.conv_state_k
            conv_state_v = rs.conv_state_v
            recurrent_state = rs.recurrent_state
            save_state = True
        else:
            conv_state_q = None
            conv_state_k = None
            conv_state_v = None
            recurrent_state = None
            save_state = False

        # cu_seqlens for variable length sequences
        cu_seqlens = params.get('cu_seqlens')

        # Q/K/V projections
        q = self.q_proj.forward(x, params)
        k = self.k_proj.forward(x, params)
        v = self.v_proj.forward(x, params)

        # Apply convolutions
        q, conv_state_q = self.q_conv1d(
            x=q,
            cache=conv_state_q,
            output_final_state=save_state,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=k,
            cache=conv_state_k,
            output_final_state=save_state,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=v,
            cache=conv_state_v,
            output_final_state=save_state,
            cu_seqlens=cu_seqlens,
        )

        # Compute decay gate g
        g = self.f_b_proj.forward(self.f_a_proj.forward(x, params), params)
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)

        # Compute beta
        beta = self.b_proj.forward(x, params).float().sigmoid()

        # Reshape for attention
        from einops import rearrange
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Choose mode based on sequence length
        mode = 'fused_recurrent' if seqlen <= 64 else 'chunk'

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        # Output gate
        gate = self.g_b_proj.forward(self.g_a_proj.forward(x, params), params)
        gate = rearrange(gate, '... (h d) -> ... h d', d=self.head_dim)

        # Apply output norm with gating
        # Note: o is shape (bsz, seqlen, num_heads, head_dim)
        # We need to normalize per head, then apply gate
        o = self.o_norm.forward(o, params, out_dtype=torch.half)
        o = o * gate.sigmoid()

        # Reshape and output projection
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj.forward(o, params)

        # Update cache
        if save_state:
            rs.conv_state_q = conv_state_q
            rs.conv_state_k = conv_state_k
            rs.conv_state_v = conv_state_v
            rs.recurrent_state = recurrent_state
            if not rs.batched:
                rs.position += seqlen
            else:
                rs.positions = [r + seqlen for r in rs.positions]

        return to2(o, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.A_log, self.key_A_log),
            (self.dt_bias, self.key_dt_bias),
        ]:
            if x is not None:
                t[k] = x
        return t


    def new_recurrent_state(self):
        return KDA_RecurrentState()


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError("KDA does not support tensor parallelism")


    def tp_export(self, plan, producer):
        raise NotImplementedError("KDA does not support tensor parallelism")


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError("KDA does not support tensor parallelism")
