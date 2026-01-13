from dataclasses import dataclass
from typing import Optional


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