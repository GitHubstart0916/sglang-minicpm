from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig

from sglang.srt.configs.mamba_utils import SimpleGLACacheParams, SimpleGLAStateShape

@dataclass
class MiniCPMSparseConfig:
    block_size: int
    dense_len: int
    init_blocks: int
    kernel_size: int
    kernel_stride: int
    topk: int
    window_size: int
    use_nope: bool

    @classmethod
    def from_hf_config(cls, hf_config) -> Optional["MiniCPMSparseConfig"]:
        """
        get MiniCPMSparseConfig from HuggingFace model config
        """
        sparse_dict = getattr(hf_config, "sparse_config", None)
        if sparse_dict is None:
            return None

        return cls(
            block_size=sparse_dict["block_size"],
            dense_len=sparse_dict["dense_len"],
            init_blocks=sparse_dict["init_blocks"],
            kernel_size=sparse_dict["kernel_size"],
            kernel_stride=sparse_dict["kernel_stride"],
            topk=sparse_dict["topk"],
            window_size=sparse_dict["window_size"],
            use_nope=sparse_dict.get("use_nope"),
        )


class MiniCPMHybridConfig(PretrainedConfig):
    """
    Configuration class for hybrid MiniCPM models.

    This config extends PretrainedConfig to match the pattern used by other
    hybrid/linear attention models (Falcon H1, Nemotron H, Kimi Linear, etc.)
    and provides cache parameters for the Simple GLA attention mechanism.
    """

    model_type = "minicpm"

    def __init__(
        self,
        # Base model config fields
        vocab_size=150528,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        intermediate_size=14336,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        # MiniCPM-specific hybrid config fields
        mixer_types=None,
        minicpm4=None,
        lightning=None,
        lightning_nh=16,
        lightning_nkv=16,
        lightning_head_dim=64,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # Hybrid config fields
        self.mixer_types = mixer_types if mixer_types is not None else None
        self.minicpm4 = minicpm4
        self.lightning = lightning
        self.lightning_nh = lightning_nh
        self.lightning_nkv = lightning_nkv
        self.lightning_head_dim = lightning_head_dim

    @property
    def mamba2_cache_params(self):
        """Return Simple GLA cache parameters for lightning attention layers."""
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if self.mixer_types is None:
            lightning_layer_ids = []
        else:
            lightning_layer_ids = [
                i for i, mixer_type in enumerate(self.mixer_types)
                if mixer_type in ["lightning", "lightning_attn", "lightning-attn"]
            ]

        if not lightning_layer_ids or not self.lightning_nkv or not self.lightning_head_dim:
            return None

        shape = SimpleGLAStateShape.create(
            tp_world_size=get_attention_tp_size(),
            num_heads=self.lightning_nkv,
            head_dim=self.lightning_head_dim,
            state_size=self.lightning_head_dim,
        )

        return SimpleGLACacheParams(shape=shape, layers=lightning_layer_ids)

    @property
    def full_attention_layer_ids(self):
        if self.mixer_types is None:
            return list(range(self.num_hidden_layers))
        else:
            return [
                i for i, mixer_type in enumerate(self.mixer_types)
                if mixer_type in ["minicpm4", "minicpm", "standard", "attention", "attn"]
            ]
