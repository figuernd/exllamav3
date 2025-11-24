from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Linear,
    BlockSparseMLP,
    GatedMLP,
)
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence
from ..modules.kimi_linear_attn import KimiMLAAttention, KimiDeltaAttention


class KimiLinearConfig(Config):
    arch_string = "KimiLinearForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": KimiLinearModel},
            **kwargs,
        )

        # Core dims
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.head_dim = self.read_cfg(int, "head_dim", self.hidden_size // self.num_q_heads)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.hidden_act = self.read_cfg(str, "hidden_act", "silu")

        # MLA specific
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.mla_use_nope = self.read_cfg(bool, "mla_use_nope", True)

        # Linear attention layout
        self.linear_attn_config = self.read_cfg(dict, "linear_attn_config", {})
        self.linear_num_heads = self.linear_attn_config.get("num_heads", self.num_q_heads)
        self.linear_head_dim = self.linear_attn_config.get("head_dim", self.head_dim)
        self.linear_short_conv_kernel = self.linear_attn_config.get("short_conv_kernel_size", 4)
        self.linear_kda_layers = set(self.linear_attn_config.get("kda_layers", []))
        self.linear_full_layers = set(self.linear_attn_config.get("full_attn_layers", []))

        # Norm / rope
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # MoE
        self.num_experts = self.read_cfg(int, "num_experts", None)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_token", None)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", None)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.moe_layer_freq = self.read_cfg(int, "moe_layer_freq", 1)
        self.moe_router_activation_func = self.read_cfg(str, "moe_router_activation_func", "sigmoid")
        self.moe_renormalize = self.read_cfg(bool, "moe_renormalize", True)
        self.use_grouped_topk = self.read_cfg(bool, "use_grouped_topk", True)
        self.num_expert_group = self.read_cfg(int, "num_expert_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)

        # Misc
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

    def is_kda_layer(self, idx: int) -> bool:
        return (idx + 1) in self.linear_kda_layers

    def is_full_attn_layer(self, idx: int) -> bool:
        return (idx + 1) in self.linear_full_layers


class KimiLinearModel(Model):
    config_class = KimiLinearConfig

    def __init__(
        self,
        config: KimiLinearConfig,
        **kwargs,
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
            block = TransformerBlock(
                config=config,
                key=f"model.layers.{idx}",
                attn_norm=RMSNorm(
                    config=config,
                    key=f"model.layers.{idx}.input_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                ),
                attn=self._build_attention(idx),
                mlp_norm=RMSNorm(
                    config=config,
                    key=f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps=config.rms_norm_eps,
                ),
                mlp=self._build_mlp(idx),
            )
            self.modules.append(block)

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
                caps={"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({"recurrent_states": True})
        self.caps.update({"supports_tp": False})

    def _build_attention(self, idx: int):
        if self.config.is_kda_layer(idx):
            return KimiDeltaAttention(
                config=self.config,
                key=f"model.layers.{idx}.self_attn",
                layer_idx=idx,
                hidden_size=self.config.hidden_size,
                conv_kernel_size=self.config.linear_short_conv_kernel,
                num_heads=self.config.linear_num_heads,
                head_dim=self.config.linear_head_dim,
            )
        else:
            return KimiMLAAttention(
                config=self.config,
                key=f"model.layers.{idx}.self_attn",
                layer_idx=idx,
                hidden_size=self.config.hidden_size,
                num_q_heads=self.config.num_q_heads,
                num_kv_heads=self.config.num_kv_heads,
                qk_rope_head_dim=self.config.qk_rope_head_dim,
                qk_nope_head_dim=self.config.qk_nope_head_dim,
                kv_lora_rank=self.config.kv_lora_rank,
                v_head_dim=self.config.v_head_dim,
                rope_settings=self.config.rope_settings,
                rms_norm_eps=self.config.rms_norm_eps,
            )

    def _build_mlp(self, idx: int):
        if (
            self.config.num_experts is not None
            and idx >= self.config.first_k_dense_replace
            and idx % max(1, self.config.moe_layer_freq) == 0
        ):
            shared_expert = None
            if self.config.num_shared_experts:
                shared_intermediate = self.config.moe_intermediate_size * self.config.num_shared_experts
                shared_expert = GatedMLP(
                    config=self.config,
                    key=f"model.layers.{idx}.mlp.shared_expert",
                    hidden_size=self.config.hidden_size,
                    intermediate_size=shared_intermediate,
                    key_up="up_proj",
                    key_gate="gate_proj",
                    key_down="down_proj",
                    qmap="block.mlp",
                    interm_dtype=torch.half,
                    out_dtype=torch.float,
                )
            router_type = "ds3" if self.config.use_grouped_topk else "std"
            return BlockSparseMLP(
                config=self.config,
                key=f"model.layers.{idx}.mlp",
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.moe_intermediate_size,
                num_experts=self.config.num_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
                key_up="experts.{expert_idx}.w3",
                key_gate="experts.{expert_idx}.w1",
                key_down="experts.{expert_idx}.w2",
                key_routing_gate="gate",
                qmap="block.mlp",
                interm_dtype=torch.half,
                out_dtype=torch.float,
                activation_fn=self.config.hidden_act,
                router_type=router_type,
                routed_scaling_factor=self.config.routed_scaling_factor,
                n_group=self.config.num_expert_group,
                topk_group=self.config.topk_group,
                shared_experts=shared_expert,
            )

        return GatedMLP(
            config=self.config,
            key=f"model.layers.{idx}.mlp",
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            key_up="up_proj",
            key_gate="gate_proj",
            key_down="down_proj",
            qmap="block.mlp",
            interm_dtype=torch.half,
            out_dtype=torch.float,
        )

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        # Rough template replicating chat_template.jinja without tool calls
        sys = system_prompt or "You are a helpful assistant provided by Moonshot-AI."
        p = ""
        p += f"<|im_system|>system<|im_middle|>{sys}<|im_end|>"
        p += f"<|im_user|>user<|im_middle|>{prompt}<|im_end|>"
        p += "<|im_assistant|>assistant<|im_middle|>"
        return p

    @override
    def check_compat(self):
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Kimi Linear requires flash-linear-attention (https://github.com/fla-org/flash-linear-attention)"
            ) from exc
