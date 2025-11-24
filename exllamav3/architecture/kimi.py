from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Linear, BlockSparseMLP, GatedMLP
from ..modules.attn import prepare_for_attn
from ..modules.mla import MLA
from ..modules.kda import KDA, prepare_for_kda_recurrence


class KimiConfig(Config):
    arch_string = "KimiLinearForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": KimiModel},
            **kwargs
        )

        # Basic params
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)

        # MLA (Full attention) params
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", None)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", None)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", None)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", None)

        # Linear attention config
        self.linear_attn_config = self.read_cfg(dict, "linear_attn_config", None)
        if self.linear_attn_config:
            self.kda_layers = self.linear_attn_config.get("kda_layers", [])
            self.full_attn_layers = self.linear_attn_config.get("full_attn_layers", [])
            self.kda_head_dim = self.linear_attn_config.get("head_dim", 128)
            self.kda_num_heads = self.linear_attn_config.get("num_heads", 32)
            self.kda_conv_kernel_size = self.linear_attn_config.get("short_conv_kernel_size", 4)
        else:
            self.kda_layers = []
            self.full_attn_layers = list(range(1, self.num_hidden_layers + 1))
            self.kda_head_dim = 128
            self.kda_num_heads = 32
            self.kda_conv_kernel_size = 4

        # MoE params
        self.num_experts = self.read_cfg(int, "num_experts", None)
        self.num_experts_per_token = self.read_cfg(int, "num_experts_per_token", None)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", None)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.moe_layer_freq = self.read_cfg(int, "moe_layer_freq", 1)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.moe_router_activation_func = self.read_cfg(str, "moe_router_activation_func", "sigmoid")
        self.num_expert_group = self.read_cfg(int, "num_expert_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)

        # MLP params
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.assert_cfg(str, "hidden_act", "silu", True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)

        # Other
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Check if layer uses KDA (linear attention)."""
        # Layers in config use 1-based indexing
        return (layer_idx + 1) in self.kda_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer uses MoE."""
        return (
            self.num_experts is not None and
            layer_idx >= self.first_k_dense_replace and
            layer_idx % self.moe_layer_freq == 0
        )


def conditional(condition, a, b):
    return a if condition else b


class KimiModel(Model):
    config_class = KimiConfig

    def __init__(
        self,
        config: KimiConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config=config,
                key="model.embed_tokens",
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            # Determine attention type
            is_kda = config.is_kda_layer(idx)
            is_moe = config.is_moe_layer(idx)

            # Create attention module
            if is_kda:
                attn = KDA(
                    config=config,
                    key=f"model.layers.{idx}.self_attn",
                    layer_idx=idx,
                    hidden_size=config.hidden_size,
                    head_dim=config.kda_head_dim,
                    num_heads=config.kda_num_heads,
                    conv_kernel_size=config.kda_conv_kernel_size,
                    rms_norm_eps=config.rms_norm_eps,
                    key_q="q_proj",
                    key_k="k_proj",
                    key_v="v_proj",
                    key_f_a="f_a_proj",
                    key_f_b="f_b_proj",
                    key_g_a="g_a_proj",
                    key_g_b="g_b_proj",
                    key_b="b_proj",
                    key_o="o_proj",
                    qmap="block.attn",
                    out_dtype=torch.float,
                )
            else:
                # MLA (full attention)
                attn = MLA(
                    config=config,
                    key=f"model.layers.{idx}.self_attn",
                    layer_idx=idx,
                    hidden_size=config.hidden_size,
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.num_kv_heads,
                    kv_lora_rank=config.kv_lora_rank,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    v_head_dim=config.v_head_dim,
                    rope_settings=config.rope_settings,
                    rms_norm_eps=config.rms_norm_eps,
                    key_q="q_proj",
                    key_kv_a="kv_a_proj_with_mqa",
                    key_kv_a_norm="kv_a_layernorm",
                    key_kv_b="kv_b_proj",
                    key_o="o_proj",
                    qmap="block.attn",
                    out_dtype=torch.float,
                )

            # Create MLP module
            if is_moe:
                mlp = BlockSparseMLP(
                    config=config,
                    key=f"model.layers.{idx}.block_sparse_moe",
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_token,
                    key_up="experts.{expert_idx}.w3",
                    key_gate="experts.{expert_idx}.w1",
                    key_down="experts.{expert_idx}.w2",
                    key_routing_gate="gate",
                    qmap="block.mlp",
                    interm_dtype=torch.half,
                    out_dtype=torch.float,
                    router_type="ds3",  # DeepSeek-V3 style with sigmoid
                    routed_scaling_factor=config.routed_scaling_factor,
                    n_group=config.num_expert_group,
                    topk_group=config.topk_group,
                    shared_experts=GatedMLP(
                        config=config,
                        key=f"model.layers.{idx}.block_sparse_moe.shared_experts",
                        hidden_size=config.hidden_size,
                        intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                        key_up="up_proj",
                        key_gate="gate_proj",
                        key_down="down_proj",
                        qmap="block.mlp",
                        interm_dtype=torch.half,
                        out_dtype=torch.float,
                    ) if config.num_shared_experts else None
                )
            else:
                mlp = GatedMLP(
                    config=config,
                    key=f"model.layers.{idx}.mlp",
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    key_up="up_proj",
                    key_gate="gate_proj",
                    key_down="down_proj",
                    qmap="block.mlp",
                    interm_dtype=torch.half,
                    out_dtype=torch.float,
                )

            self.modules.append(
                TransformerBlock(
                    config=config,
                    key=f"model.layers.{idx}",
                    attn_norm=RMSNorm(
                        config=config,
                        key=f"model.layers.{idx}.input_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    attn=attn,
                    mlp_norm=RMSNorm(
                        config=config,
                        key=f"model.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    mlp=mlp,
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "model.embed_tokens"

        self.modules += [
            RMSNorm(
                config=config,
                key="model.norm",
                rms_norm_eps=config.rms_norm_eps,
                out_dtype=torch.half,
            ),
            Linear(
                config=config,
                key="lm_head",
                qbits_key="head_bits",
                alt_key=head_alt_key,
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                qmap="block",
                caps={"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Mark that we need recurrent cache for KDA layers
        if config.kda_layers:
            self.caps.update({"recurrent_states": True})

        # TP not supported for hybrid attention models
        self.caps.update({"supports_tp": False})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        if self.config.kda_layers:
            prepare_for_kda_recurrence(input_ids, params, self)
        return input_ids


    def get_kda_layers(self):
        """Get all KDA attention modules."""
        kda_modules = []
        for module in self.modules:
            if isinstance(module, TransformerBlock):
                if isinstance(module.attn, KDA):
                    kda_modules.append(module.attn)
        return kda_modules


    @override
    def check_compat(self):
        try:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
            from fla.ops.kda.gate import fused_kda_gate
            from fla.modules import FusedRMSNormGated, ShortConvolution
        except ModuleNotFoundError as e:
            print(" ## Kimi requires fla-core (https://github.com/fla-org/flash-linear-attention)")
            print(" ## Install with: pip install -U fla-core")
            raise e
