from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from ..cache.recurrent import CacheableState
from . import Module, Linear, RMSNorm

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ModuleNotFoundError:
    chunk_kda = None
    fused_recurrent_kda = None
    fused_kda_gate = None
from ..model.config import Config
from ..util.tensor import to2


class KimiMLAAttention(Module):
    """
    Multi-latent attention used by Kimi Linear. Implements the LoRA-style KV factorization
    and mixes NoPE/RoPE heads before delegating to FlashAttention.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        rope_settings,
        rms_norm_eps: float,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, "block.attn")
        assert num_q_heads == num_kv_heads, "Kimi MLA expects MQA-style heads."

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self.head_dim = self.q_head_dim
        self.num_q_heads = self.num_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.scale = self.q_head_dim ** -0.5
        self.sliding_window = -1

        self.q_proj = Linear(
            config=config,
            key=f"{key}.q_proj",
            qbits_key="q",
            in_features=hidden_size,
            out_features=self.num_heads * self.q_head_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.register_submodule(self.q_proj)

        self.kv_a_proj = Linear(
            config=config,
            key=f"{key}.kv_a_proj_with_mqa",
            in_features=hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
            pad_to=64,
        )
        self.register_submodule(self.kv_a_proj)

        self.kv_a_norm = RMSNorm(
            config=config,
            key=f"{key}.kv_a_layernorm",
            rms_norm_eps=rms_norm_eps,
            out_dtype=torch.half,
        )
        self.register_submodule(self.kv_a_norm)

        self.kv_b_proj = Linear(
            config=config,
            key=f"{key}.kv_b_proj",
            in_features=self.kv_lora_rank,
            out_features=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.register_submodule(self.kv_b_proj)

        self.o_proj = Linear(
            config=config,
            key=f"{key}.o_proj",
            in_features=self.num_heads * self.v_head_dim,
            out_features=hidden_size,
            qmap="block.attn.o",
            out_dtype=self.out_dtype,
            qbits_mod_key="o",
        )
        self.register_submodule(self.o_proj)

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.has_split_cache = False
        self.tp_reduce = False

        self.caps.update({"kv_cache": True})

    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.kv_a_proj.optimizer_targets()
        v = self.kv_b_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)

    def unload(self):
        super().unload()

    def _reshape_heads(self, tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
        bsz, seqlen, _ = tensor.shape
        return tensor.view(bsz, seqlen, self.num_heads, head_dim)

    def _maybe_pad_value(self, value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if self.v_head_dim == self.q_head_dim:
            return value, False
        pad = self.q_head_dim - self.v_head_dim
        zeros = torch.zeros(
            (*value.shape[:-1], pad),
            dtype=value.dtype,
            device=value.device,
        )
        padded = torch.cat([value, zeros], dim=-1)
        return padded, True

    def _project_qkv(self, x: torch.Tensor, params: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj.forward(x, params)
        bsz, seqlen, _ = q.shape
        q = q.view(bsz, seqlen, self.num_heads, self.q_head_dim)
        q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_combined = self.kv_a_proj.forward(x, params)
        k_pass, k_rot = torch.split(kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_norm.forward(k_pass, params, out_dtype=torch.half)
        kv = self.kv_b_proj.forward(k_pass, params)
        kv = kv.view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_linear, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(bsz, seqlen, 1, self.qk_rope_head_dim).expand(-1, -1, self.num_heads, -1)

        query = torch.cat([q_pass, q_rot], dim=-1)
        key = torch.cat([k_linear, k_rot], dim=-1)

        query = query.reshape(bsz, seqlen, self.num_heads * self.q_head_dim)
        key = key.reshape(bsz, seqlen, self.num_heads * self.q_head_dim)
        value = value.reshape(bsz, seqlen, self.num_heads * self.v_head_dim)
        return query, key, value

    def _flash_attn_nc(self, x: torch.Tensor, bsz: int, seqlen: int, params: dict) -> torch.Tensor:
        q, k, v = self._project_qkv(x, params)
        q = self._reshape_heads(q, self.q_head_dim)
        k = self._reshape_heads(k, self.q_head_dim)
        v = self._reshape_heads(v, self.v_head_dim)
        v_padded, padded = self._maybe_pad_value(v)

        causal = params.get("causal", True)
        training = params.get("training", False)
        dropout = 0.0 if not training else self.attention_dropout
        out = flash_attn_func(
            q,
            k,
            v_padded,
            dropout_p=dropout,
            softmax_scale=self.scale,
            causal=causal,
        )
        if padded:
            out = out[:, :, :, : self.v_head_dim]
        out = out.reshape(bsz, seqlen, self.num_heads * self.v_head_dim)
        out = self.o_proj.forward(out, params)
        return out

    def _flash_attn_cached(self, x: torch.Tensor, bsz: int, seqlen: int, params: dict) -> torch.Tensor:
        cache = params.get("cache")
        assert cache is not None, "Paged cache required for flash_attn mode."
        block_table = params.get("block_table")
        cache_seqlens = params.get("cache_seqlens")
        causal = params.get("causal", True)

        q, k, v = self._project_qkv(x, params)
        q = self._reshape_heads(q, self.q_head_dim)
        k = self._reshape_heads(k, self.q_head_dim)
        v = self._reshape_heads(v, self.v_head_dim)
        v_padded, padded = self._maybe_pad_value(v)

        cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)

        out = flash_attn_with_kvcache(
            q=q,
            k=k,
            v=v_padded,
            k_cache=cache_k,
            v_cache=cache_v,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            causal=causal,
            softmax_scale=self.scale,
            window_size=(self.sliding_window, self.sliding_window),
        )
        cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)

        if padded:
            out = out[:, :, :, : self.v_head_dim]
        out = out.reshape(bsz, seqlen, self.num_heads * self.v_head_dim)
        out = self.o_proj.forward(out, params)
        return out

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")
        if attn_mode == "flash_attn":
            out = self._flash_attn_cached(x, bsz, seqlen, params)
        elif attn_mode == "flash_attn_nc":
            out = self._flash_attn_nc(x, bsz, seqlen, params)
        else:
            raise ValueError(f"Unsupported attention mode {attn_mode} for Kimi MLA.")
        return to2(out, out_dtype, self.out_dtype)


class KimiRecurrentState(CacheableState):

    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        conv_q: torch.Tensor | None = None,
        conv_k: torch.Tensor | None = None,
        conv_v: torch.Tensor | None = None,
        recurrent: torch.Tensor | None = None,
        batched: bool = False,
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.conv_q = conv_q
        self.conv_k = conv_k
        self.conv_v = conv_v
        self.recurrent = recurrent
        self.batched = batched

    @override
    def stash(self):
        def _cpu(t): return t.cpu() if t is not None else None
        return KimiRecurrentState(
            self.position,
            self.positions,
            _cpu(self.conv_q),
            _cpu(self.conv_k),
            _cpu(self.conv_v),
            _cpu(self.recurrent),
            self.batched,
        )

    @override
    def unstash(self, device):
        def _to(t):
            return t.to(device, non_blocking=True) if t is not None else None
        return KimiRecurrentState(
            self.position,
            self.positions,
            _to(self.conv_q),
            _to(self.conv_k),
            _to(self.conv_v),
            _to(self.recurrent),
            self.batched,
        )

    @override
    def get_size(self):
        total = 0
        for tensor in (self.conv_q, self.conv_k, self.conv_v, self.recurrent):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    def collect_batch(self, batch: list["KimiRecurrentState"]):
        def _cat(attr):
            tensors = [getattr(b, attr) for b in batch]
            if tensors[0] is None:
                return None
            return torch.cat(tensors, dim=0)
        positions = [b.position or 0 for b in batch]
        return KimiRecurrentState(
            None,
            positions,
            _cat("conv_q"),
            _cat("conv_k"),
            _cat("conv_v"),
            _cat("recurrent"),
            True,
        )

    def distribute_batch(self, batch: list["KimiRecurrentState"]):
        for idx, target in enumerate(batch):
            for attr in ("conv_q", "conv_k", "conv_v", "recurrent"):
                tensor = getattr(self, attr)
                if tensor is None:
                    continue
                slice_tensor = tensor[idx : idx + 1, ...]
                current = getattr(target, attr)
                if current is None:
                    setattr(target, attr, slice_tensor.clone())
                else:
                    current.copy_(slice_tensor)
            target.position = self.positions[idx]


class KimiDeltaAttention(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        conv_kernel_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__(config, key, "block.attn")
        if chunk_kda is None or fused_recurrent_kda is None or fused_kda_gate is None:
            raise RuntimeError(
                "flash-linear-attention is required for Kimi Delta attention."
            )
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.rms_norm_eps = config.rms_norm_eps

        self.q_proj = Linear(
            config=config,
            key=f"{key}.q_proj",
            in_features=hidden_size,
            out_features=self.proj_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.k_proj = Linear(
            config=config,
            key=f"{key}.k_proj",
            in_features=hidden_size,
            out_features=self.proj_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.v_proj = Linear(
            config=config,
            key=f"{key}.v_proj",
            in_features=hidden_size,
            out_features=self.proj_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.f_a_proj = Linear(
            config=config,
            key=f"{key}.f_a_proj",
            in_features=hidden_size,
            out_features=head_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.f_b_proj = Linear(
            config=config,
            key=f"{key}.f_b_proj",
            in_features=head_dim,
            out_features=self.proj_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.g_a_proj = Linear(
            config=config,
            key=f"{key}.g_a_proj",
            in_features=hidden_size,
            out_features=head_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.g_b_proj = Linear(
            config=config,
            key=f"{key}.g_b_proj",
            in_features=head_dim,
            out_features=self.proj_dim,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.b_proj = Linear(
            config=config,
            key=f"{key}.b_proj",
            in_features=hidden_size,
            out_features=num_heads,
            qmap="block.attn.input",
            out_dtype=torch.float,
        )
        self.o_proj = Linear(
            config=config,
            key=f"{key}.o_proj",
            in_features=self.proj_dim,
            out_features=hidden_size,
            qmap="block.attn.output",
            out_dtype=torch.float,
        )

        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.f_a_proj,
            self.f_b_proj,
            self.g_a_proj,
            self.g_b_proj,
            self.b_proj,
            self.o_proj,
        ]:
            self.register_submodule(module)

        self.A_log_key = f"{key}.A_log"
        self.dt_bias_key = f"{key}.dt_bias"
        self.conv_keys = {
            "q": (f"{key}.q_conv1d.weight", f"{key}.q_conv1d.bias"),
            "k": (f"{key}.k_conv1d.weight", f"{key}.k_conv1d.bias"),
            "v": (f"{key}.v_conv1d.weight", f"{key}.v_conv1d.bias"),
        }
        self.o_norm_weight_key = f"{key}.o_norm.weight"

        self.A_log = None
        self.dt_bias = None
        self.conv_weights = {"q": None, "k": None, "v": None}
        self.conv_biases = {"q": None, "k": None, "v": None}
        self.o_norm_weight = None

        self.caps.update({"recurrent_cache": True})

    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.A_log = stc.get_tensor(self.A_log_key, device, allow_bf16=True)
        self.dt_bias = stc.get_tensor(self.dt_bias_key, device, allow_bf16=True)
        for name in ("q", "k", "v"):
            w_key, b_key = self.conv_keys[name]
            self.conv_weights[name] = stc.get_tensor(w_key, device, allow_bf16=True)
            self.conv_biases[name] = stc.get_tensor(b_key, device, allow_bf16=True, optional=True)
        self.o_norm_weight = stc.get_tensor(self.o_norm_weight_key, device, allow_bf16=True)

    def unload(self):
        self.A_log = None
        self.dt_bias = None
        self.o_norm_weight = None
        for name in ("q", "k", "v"):
            self.conv_weights[name] = None
            self.conv_biases[name] = None
        super().unload()

    def new_recurrent_state(self):
        return KimiRecurrentState()

    def _alloc_conv_state(self, batch: int, dtype: torch.dtype, device: torch.device):
        return torch.zeros((batch, self.proj_dim, self.conv_kernel_size), dtype=dtype, device=device)

    def _short_conv(self, tensor, name, state):
        weight = self.conv_weights[name]
        bias = self.conv_biases[name]
        bsz, seqlen, _ = tensor.shape
        xt = tensor.transpose(1, 2)
        if state is None:
            state = self._alloc_conv_state(bsz, xt.dtype, tensor.device)
        y = torch.cat([state, xt], dim=-1).to(weight.dtype)
        new_state = y[:, :, -self.conv_kernel_size:].contiguous()
        out = F.conv1d(y, weight, bias, padding=0, groups=self.proj_dim)
        out = F.silu(out[:, :, -seqlen:]).transpose(1, 2).to(tensor.dtype)
        return out, new_state

    def _apply_o_norm(self, tensor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        y = tensor.to(torch.float32)
        var = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(var + self.rms_norm_eps)
        y = y * torch.sigmoid(gate.to(torch.float32))
        y = y * self.o_norm_weight
        return y.to(tensor.dtype)

    def _cu_seqlens(self, batch: int, seqlen: int, device: torch.device):
        return torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        rs_map = params.get("recurrent_states")
        if rs_map is None:
            raise ValueError("Kimi Delta attention requires recurrent_states parameter.")
        state = rs_map[self.layer_idx]

        q = self.q_proj.forward(x, params)
        k = self.k_proj.forward(x, params)
        v = self.v_proj.forward(x, params)

        q, state.conv_q = self._short_conv(q, "q", state.conv_q)
        k, state.conv_k = self._short_conv(k, "k", state.conv_k)
        v, state.conv_v = self._short_conv(v, "v", state.conv_v)

        g = self.f_b_proj.forward(self.f_a_proj.forward(x, params), params)
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = self.b_proj.forward(x, params).float().sigmoid()

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim)
        g = g.view(bsz, seqlen, self.num_heads, self.head_dim)
        beta = beta.view(bsz, seqlen, self.num_heads)

        if state.recurrent is None:
            state.recurrent = torch.zeros(
                (bsz, self.num_heads, self.head_dim, self.head_dim),
                dtype=torch.float32,
                device=x.device,
            )

        cu = self._cu_seqlens(bsz, seqlen, x.device)
        if seqlen <= 64:
            attn_out, new_recurrent = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state.recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
            )
        else:
            attn_out, new_recurrent = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state.recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
            )

        gate = self.g_b_proj.forward(self.g_a_proj.forward(x, params), params)
        gate = gate.view(bsz, seqlen, self.num_heads, self.head_dim)
        attn_out = self._apply_o_norm(attn_out, gate)
        attn_out = attn_out.reshape(bsz, seqlen, self.num_heads * self.head_dim)
        attn_out = self.o_proj.forward(attn_out, params)

        state.recurrent = new_recurrent
        if not state.batched:
            state.position = (state.position or 0) + seqlen
        else:
            state.positions = [pos + seqlen for pos in state.positions]

        return to2(attn_out, out_dtype, self.out_dtype)

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        conv_kernel_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__(config, key, "block.attn")
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.caps.update({"recurrent_cache": True})

    @override
    def optimizer_targets(self):
        return []

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        raise NotImplementedError("Kimi Delta attention not implemented yet.")
