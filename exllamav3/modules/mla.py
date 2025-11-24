from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from ..constants import PAGE_SIZE
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from ..model.model_tp_alloc import TPAllocation

"""
Multi-Latent Attention (MLA) module for Kimi architecture.

This is a compressed KV attention mechanism based on DeepSeek-V3:
- Q projection outputs full heads with rope and nope components
- KV is compressed via kv_a_proj_with_mqa to (kv_lora_rank + qk_rope_head_dim)
- Compressed KV is normalized then expanded via kv_b_proj
- The expanded KV is split into k_nope, k_rope, and v components
"""


class MLA(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        rope_settings: RopeSettings | None,
        rms_norm_eps: float = 1e-6,
        key_q: str | None = None,
        key_kv_a: str | None = None,
        key_kv_a_norm: str | None = None,
        key_kv_b: str | None = None,
        key_o: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.rms_norm_eps = rms_norm_eps

        # Scaling factor
        self.sm_scale = self.q_head_dim ** (-0.5)

        # Q projection: outputs (num_q_heads * q_head_dim)
        self.q_proj = Linear(
            config, f"{key}.{key_q}",
            hidden_size,
            num_q_heads * self.q_head_dim,
            qmap=qmap + ".input" if qmap else None
        )
        self.register_submodule(self.q_proj)

        # Compressed KV projection: outputs (kv_lora_rank + qk_rope_head_dim)
        self.kv_a_proj = Linear(
            config, f"{key}.{key_kv_a}",
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            qmap=qmap + ".input" if qmap else None
        )
        self.register_submodule(self.kv_a_proj)

        # LayerNorm on compressed KV (applied to kv_lora_rank portion)
        self.kv_a_norm = RMSNorm(
            config, f"{key}.{key_kv_a_norm}",
            rms_norm_eps=rms_norm_eps
        )
        self.register_submodule(self.kv_a_norm)

        # KV expansion: from kv_lora_rank to heads * (qk_nope_head_dim + v_head_dim)
        self.kv_b_proj = Linear(
            config, f"{key}.{key_kv_b}",
            kv_lora_rank,
            num_q_heads * (qk_nope_head_dim + v_head_dim),
            qmap=qmap + ".input" if qmap else None
        )
        self.register_submodule(self.kv_b_proj)

        # Output projection
        self.o_proj = Linear(
            config, f"{key}.{key_o}",
            num_q_heads * v_head_dim,
            hidden_size,
            qmap=qmap + ".o" if qmap else None,
            out_dtype=out_dtype
        )
        self.register_submodule(self.o_proj)

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        kv_a = self.kv_a_proj.optimizer_targets()
        kv_b = self.kv_b_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, kv_a + kv_b, o]]


    def load_local(self, device, **kwargs):
        # Cache
        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            # MLA only applies RoPE to qk_rope_head_dim portion
            # Create custom rope settings with correct rotary dimension
            from dataclasses import replace
            mla_rope_settings = replace(
                self.rope_settings,
                rotary_dim=self.qk_rope_head_dim,
                head_dim=self.qk_rope_head_dim
            )
            self.rope = RoPE(
                device,
                mla_rope_settings,
            )


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.rope = None


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")

        match attn_mode:
            case "flash_attn":
                x = self.decode_flash_attn(x, bsz, seqlen, params)
            case "flash_attn_nc":
                x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"MLA: Unknown attn_mode: {attn_mode}")

        if self.tp_reduce:
            params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    def project_qkv(self, x: torch.Tensor, params: dict):
        """
        Project input to Q, K, V using MLA's compressed representation.
        """
        bsz, seq_len, _ = x.shape

        # Q projection and split into nope and rope parts
        q = self.q_proj.forward(x, params)
        q = q.view(bsz, seq_len, self.num_q_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Compressed KV projection
        compressed_kv = self.kv_a_proj.forward(x, params)
        k_compressed, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Normalize and expand compressed KV
        k_compressed = self.kv_a_norm.forward(k_compressed, params)
        kv_expanded = self.kv_b_proj.forward(k_compressed, params)
        kv_expanded = kv_expanded.view(
            bsz, seq_len, self.num_q_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Expand k_rope from (bsz, seq_len, rope_dim) to (bsz, seq_len, heads, rope_dim)
        k_rope = k_rope.view(bsz, seq_len, 1, self.qk_rope_head_dim)
        k_rope = k_rope.expand(bsz, seq_len, self.num_q_heads, self.qk_rope_head_dim)

        return q_nope, q_rope, k_nope, k_rope, v


    def decode_flash_attn_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        causal = params.get("causal", True)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)

        q_nope, q_rope, k_nope, k_rope, v = self.project_qkv(x, params)

        # Apply RoPE to rope components only
        if self.rope:
            q_rope, k_rope = self.rope.apply(
                q_rope, k_rope,
                position,
                positions,
                position_ids,
                True,
                None,
                None,
                self.rms_norm_eps,
                0.0,
                inv_freq,
            )

        # Concatenate nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Pad v to match q head dim for flash attention if needed
        if self.q_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.q_head_dim - self.v_head_dim])

        o = flash_attn_func(
            q=q,
            k=k,
            v=v,
            causal=causal,
            softmax_scale=self.sm_scale,
        )

        # Remove padding from output
        if self.q_head_dim != self.v_head_dim:
            o = o[:, :, :, :self.v_head_dim]

        o = o.reshape(bsz, seqlen, self.num_q_heads * self.v_head_dim)
        o = self.o_proj.forward(o, params)
        return o


    def decode_flash_attn(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)

        q_nope, q_rope, k_nope, k_rope, v = self.project_qkv(x, params)

        # Apply RoPE to rope components only
        if self.rope:
            q_rope, k_rope = self.rope.apply(
                q_rope, k_rope,
                position,
                positions,
                position_ids,
                True,
                None,
                None,
                self.rms_norm_eps,
                0.0,
                inv_freq,
            )

        # Concatenate nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Pad v to match q head dim for flash attention if needed
        if self.q_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.q_head_dim - self.v_head_dim])

        if self.has_split_cache:
            cache_k, cache_v = self.tp_cache_lookup[cache].get_kv(cache_seqlens, block_table)
        else:
            cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)

        o = flash_attn_with_kvcache(
            q=q,
            k=k,
            v=v,
            k_cache=cache_k,
            v_cache=cache_v,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            causal=causal,
            softmax_scale=self.sm_scale,
        )

        if self.has_split_cache:
            self.tp_cache_lookup[cache].update_kv(cache_seqlens, block_table, cache_k, cache_v, seqlen)
        else:
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)

        # Remove padding from output
        if self.q_head_dim != self.v_head_dim:
            o = o[:, :, :, :self.v_head_dim]

        o = o.reshape(bsz, seqlen, self.num_q_heads * self.v_head_dim)
        o = self.o_proj.forward(o, params)
        return o


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        # TP not supported for MLA yet
        raise NotImplementedError("MLA does not support tensor parallelism")


    def tp_export(self, plan, producer):
        raise NotImplementedError("MLA does not support tensor parallelism")


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError("MLA does not support tensor parallelism")
