from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var
from sglang.srt.distributed import get_tensor_model_parallel_world_size

from sglang.srt.layers.attention.minicpm_sparse_utils import (
    CompressionLevelMetadata,
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadataBuilder,
    allocate_and_compress_keys,
    compressed_attention,
    get_compress_k_v2,
    get_compress_k_v2_padded,
)

from sglang.srt.layers.attention.minicpm_sparse_kernels import get_sparse_block_table

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


logger = logging.getLogger(__name__)


@dataclass
class MiniCPMBackendMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    page_table: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Stage1 optimization metadata
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: Optional[int] = None
    cache_seqlens_int32_stage1: torch.Tensor = None
    
class AscendAttnMaskBuilder:
    def __init__(self, model_runner: ModelRunner, device, use_fia, use_mla):
        """
        Initialize the AscendAttnMaskBuilder class.

        :param model_runner: ModelRunner instance for model execution.
        :param device: Device to run the model on (e.g., 'cuda', 'npu').
        :param use_fia: Boolean flag to indicate if environment variable ASCEND_USE_FIA is set to 1.
        """
        self.use_fia = use_fia
        self.model_runner = model_runner
        self.device = device

        # Initialize mask
        mask_len = 128
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(
            self.device
        )

        # Initialize FIA mask
        fia_mask_len = 2048
        self.fia_mask = self.generate_mask_flag(fia_mask_len).to(self.device)

        # Initialize MTP mask
        mtp_mask_len = 2048
        self.mtp_mask = self.generate_mask_flag(mtp_mask_len).to(self.device)

        # Initialize mixed chunk mask cache
        mixed_mask_len = 2048
        self.mixed_chunk_attn_mask = self.get_splitfuse_attn_mask(mixed_mask_len)

        if use_mla:
            # Initialize RingMla mask
            ringmla_mask_len = 512
            self.ringmla_mask = self.generate_attn_mask(
                ringmla_mask_len, "norm", torch.bfloat16
            ).to(self.device)

    @staticmethod
    def generate_mask_flag(max_seq_len):
        """
        Generate a mask flag for attention masks.

        :param max_seq_len: Maximum sequence length for the mask.
        :return: A boolean tensor representing the mask flag.
        """
        # Construct lower triangle matrix.
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Create upper triangle matrix used to mark mask positions.
        mask_flag = ~mask_flag
        return mask_flag

    @staticmethod
    def generate_attn_mask(max_seq_len, mode, dtype=torch.float16):
        """
        Generate an attention mask.

        :param max_seq_len: Maximum sequence length for the mask.
        :param mode: Mode of the mask ('mix' or 'norm').
        :param dtype: Data type of the mask tensor.
        :return: A tensor representing the attention mask.
        """
        mask_flag = AscendAttnMaskBuilder.generate_mask_flag(max_seq_len)
        if mode == "mix":
            mask_value = (
                float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
            )
        else:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        attn_mask = (
            torch.zeros(size=(max_seq_len, max_seq_len))
            .masked_fill_(mask_flag, mask_value)
            .to(dtype)
        )
        return attn_mask

    @staticmethod
    def get_attention_mask_id(seq_lens, extend_lens):
        """
        Generate attention mask IDs based on sequence lengths and extended lengths.

        :param seq_lens: Sequence lengths.
        :param extend_lens: Extended lengths.
        :return: A tensor containing the attention mask IDs.
        """
        starts = seq_lens - extend_lens
        ends = seq_lens

        # Use torch.stack to stack the start and end indices together
        ranges = torch.stack((starts, ends), dim=-1)

        # Use list comprehension to generate tensors for each range and concatenate them
        attn_mask_id = torch.cat([torch.arange(start, end) for start, end in ranges])
        return attn_mask_id

    def update_attn_cache(
        self,
        seqlen: int,
        mask_cache: torch.Tensor,
        seq_len_cached: int,
        dtype: torch.dtype,
        mode,
    ):
        """
        Update the attention mask cache.

        :param seqlen: Maximum sequence length.
        :param mask_cache: Current attention mask cache.
        :param seq_len_cached: Cached sequence length.
        :param dtype: Data type of the mask tensor.
        :param mode: Mode of the mask ('mix' or 'norm').
        :return: Updated mask cache and sequence length cache.
        """
        if seqlen > seq_len_cached:
            seq_len_cached = seqlen
            mask_cache = self.generate_attn_mask(seqlen, mode, dtype)
        if mask_cache.dtype != dtype:
            mask_cache = mask_cache.to(dtype)
        return mask_cache, seq_len_cached

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :return: A tensor representing the splitfuse attention mask.
        """
        attn_mask = (
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1)
            .to(torch.int8)
            .to(self.device)
        )
        return attn_mask


class AscendMiniCPMSparseBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.use_mla = False
        self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_sparse_k1_token = (
            model_runner.req_to_token_pool.req_to_sparse_k1_token
        )
        self.req_to_sparse_k2_token = (
            model_runner.req_to_token_pool.req_to_sparse_k2_token
        )
        tp_size = get_tensor_model_parallel_world_size()
        self.num_kv_heads = model_runner.model_config.num_key_value_heads // tp_size
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.ascend_attn_mask_builder = AscendAttnMaskBuilder(
            model_runner, self.device, self.use_fia, self.use_mla
        )
        self.mask, self.fia_mask, self.mtp_mask, self.mix_mask = (
            self.ascend_attn_mask_builder.mask,
            self.ascend_attn_mask_builder.fia_mask,
            self.ascend_attn_mask_builder.mtp_mask,
            self.ascend_attn_mask_builder.mixed_chunk_attn_mask,
        )
        if self.use_mla:
            raise ValueError("MLA is not supported in AscendMiniCPMSparseBackend.")
        # Sparse attention configuration (required for MiniCPM)
        hf_config = getattr(model_runner.model_config, "hf_config", None)
        self.has_sparse_attention = hf_config is not None and getattr(
            hf_config, "has_sparse_attention", False
        )
        # MiniCPM must have sparse attention enabled
        if not self.has_sparse_attention:
            raise ValueError(
                "MiniCPM model must have sparse attention enabled. "
                "Please ensure the model config has 'has_sparse_attention=True'."
            )
        self.kernel_size = hf_config.sparse_kernel_size
        self.kernel_stride = hf_config.sparse_kernel_stride
        self.init_blocks = hf_config.sparse_init_blocks
        self.block_size = hf_config.sparse_block_size
        self.window_size = hf_config.sparse_window_size
        self.dense_as_sparse = model_runner.server_args.dense_as_sparse
        self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
        self.config_dense_len = hf_config.sparse_dense_len
        topk = hf_config.sparse_topk
        self.use_nope = hf_config.sparse_use_nope
        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.sparse_topk = topk + (self.window_size // self.block_size)
        self.num_sparse_topk_tokens = self.block_size * self.sparse_topk
        self.num_sparse_topk_blocks = self.num_sparse_topk_tokens // self.page_size

        # Head group number derived from model configuration
        self.head_dim = model_runner.model_config.head_dim
        self.head_group_num = model_runner.model_config.num_key_value_heads
        self.heads_per_group = (
            model_runner.model_config.num_attention_heads // self.head_group_num
        )
        self.k1_kernel_size = self.kernel_size
        self.k1_kernel_stride = self.kernel_stride
        self.k2_kernel_size = self.kernel_size * 4
        self.k2_kernel_stride = self.kernel_stride * 4

        # fuse_topk is not supported on Ascend
        self.fuse_topk = False
        self.split_stage1 = model_runner.server_args.split_stage1

        # Initialize sparse attention helpers (required for MiniCPM)
        sparse_config = SparseConfig.from_model_config(
            hf_config, model_runner.model_config
        )
        self.sparse_batch_analyzer = SparseBatchAnalyzer(sparse_config)
        self.sparse_metadata_builder = SparseMetadataBuilder(
            sparse_config,
            num_kv_heads=self.num_kv_heads,
            max_context_len=self.max_context_len,
        )
 
    def update_batch_for_sparse(
        self, forward_batch: ForwardBatch, metadata: MiniCPMBackendMetadata
    ):
        cu_seqlens_q = metadata.cu_seqlens_q

        compression_metadata = (
            self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                req_to_sparse_k1_token=self.req_to_sparse_k1_token,
                req_to_sparse_k2_token=self.req_to_sparse_k2_token,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                cu_seqlens_q=cu_seqlens_q,
            )
        )

        # Map k1/k2 compression metadata objects
        metadata.k1 = compression_metadata["k1"]
        metadata.k2 = compression_metadata["k2"]

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.sparse_bs_list = (
                self.sparse_batch_analyzer.identify_sparse_batches(forward_batch, self.dense_as_sparse)
            )

            seqlen_q_sparse_bs, metadata.seqlen_k_sparse_bs_tensor = (
                self.sparse_metadata_builder.build_sequence_lengths(
                    cu_seqlens_q,
                    forward_batch.extend_prefix_lens,
                    metadata.sparse_bs_list,
                )
            )

            cu_seqlens_q_sparse_bs = torch.tensor(
                [0] + seqlen_q_sparse_bs, dtype=torch.int32, device=cu_seqlens_q.device
            ).cumsum(dtype=torch.int32, dim=0)

            extend_prefix_lens_sparse = torch.tensor(
                [
                    forward_batch.extend_prefix_lens_cpu[bs]
                    for bs in metadata.sparse_bs_list
                ],
                dtype=torch.long,
                device="cpu",
            )

            metadata.token_to_bs, metadata.token_pos_in_bs = (
                self.sparse_metadata_builder.build_token_mappings(
                    cu_seqlens_q_sparse_bs,
                    extend_prefix_lens_sparse,
                    seqlen_q_sparse_bs,
                )
            )
            metadata.token_to_bs = metadata.token_to_bs.to(device=metadata.cu_seqlens_q.device)
            metadata.token_pos_in_bs = metadata.token_pos_in_bs.to(device=metadata.cu_seqlens_q.device)

            prefill_metadata = (
                self.sparse_metadata_builder.build_sparse_prefill_metadata(
                    forward_batch=forward_batch,
                    base_metadata=metadata,
                    sparse_bs_list=metadata.sparse_bs_list,
                    head_group_num=self.head_group_num,
                    dense_len=self.dense_len,
                    sparse_topk=self.sparse_topk,
                    sparse_block_size=self.block_size,
                    page_size=self.page_size,
                    cu_seqlens_q=cu_seqlens_q,
                    sparse_page_table_dtype=metadata.page_table.dtype,
                    sparse_page_table_device=metadata.page_table.device,
                )
            )

            metadata.sparse_page_table = prefill_metadata["sparse_page_table"]
            metadata.sparse_cu_seqlens_q_cpu = prefill_metadata[
                "sparse_cu_seqlens_q_cpu"
            ]
            metadata.sparse_cu_seqlens_q = prefill_metadata["sparse_cu_seqlens_q"]
            metadata.old_bs_to_new_bs_range = prefill_metadata["old_bs_to_new_bs_range"]
            metadata.sparse_max_seq_len_q = prefill_metadata["sparse_max_seq_len_q"]

            forward_batch.sparse_batch_size = len(metadata.sparse_bs_list)
            forward_batch.sparse_idx = prefill_metadata["sparse_idx"]

            # Stage1 optimization metadata for prefill mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            seqlens_q_sparse_list = []
            for i in range(forward_batch.batch_size):
                if forward_batch.seq_lens_cpu[i] >= self.dense_len:
                    seqlens_q_sparse_list.append(forward_batch.extend_seq_lens_cpu[i])
            
            seqlen_q_sparse_tensor = torch.tensor(seqlens_q_sparse_list, dtype=torch.int32, device=metadata.cu_seqlens_q.device)
            cu_seqlen_q_sparse_tensor = F.pad(torch.cumsum(seqlen_q_sparse_tensor, dim=0, dtype=torch.int32), (1, 0))
            metadata.cu_seqlens_q_adjusted = cu_seqlen_q_sparse_tensor * self.heads_per_group
            metadata.max_seqlen_q_adjusted = seqlen_q_sparse_tensor.max().item() * self.heads_per_group
        else:
            decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                sparse_block_size=self.block_size,
                page_size=self.page_size,
            )

            metadata.sparse_cache_seqlens_int32 = decode_metadata[
                "sparse_cache_seqlens_int32"
            ]
            metadata.sparse_cu_seqlens_k = decode_metadata["sparse_cu_seqlens_k"]
            metadata.sparse_cu_seqlens_q = decode_metadata["sparse_cu_seqlens_q"]
            metadata.sparse_page_table = decode_metadata["sparse_page_table"]
            metadata.token_to_bs = decode_metadata["token_to_bs"]

            # Stage1 optimization metadata for decode mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            metadata.cu_seqlens_q_adjusted = metadata.cu_seqlens_q * self.heads_per_group
            metadata.max_seqlen_q_adjusted = metadata.max_seq_len_q * self.heads_per_group
        
    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = MiniCPMBackendMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        seq_lens_max = seqlens_in_batch.max()
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device
    
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.page_table = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens

        if (
            self.use_mla
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            and not forward_batch.forward_mode.is_target_verify()
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        ):
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to(
                "cpu"
            )
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(
                forward_batch.req_pool_indices.tolist(), seq_prefix_lens
            ):
                req_indices = forward_batch.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        self.graph_mode = False

        # For MiniCPM
        self.forward_metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
        self.forward_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            self.forward_metadata.max_seq_len_q = 1
            self.forward_metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            self.forward_metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            self.forward_metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                self.forward_metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                self.forward_metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                self.forward_metadata.max_seq_len_q = self.forward_metadata.max_seq_len_k
                self.forward_metadata.cu_seqlens_q = self.forward_metadata.cu_seqlens_k

            self.forward_metadata.sparse_cache_seqlens_int32 = forward_batch.sparse_cache_seqlens_int32_cpu.to(device=device)
            self.forward_metadata.sparse_cu_seqlens_k = forward_batch.sparse_cu_seqlens_k_cpu.to(device=device)

        self.update_batch_for_sparse(forward_batch, self.forward_metadata)

    def get_topk_for_sparse(
        self,
        query_states,
        key_states,
        value_states,
        query_length,
        layer,
        forward_batch,
        is_prefill=True,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
        decode_batch_id=0,
    ):
        if is_prefill:

            all_sparse = forward_batch.sparse_batch_size == forward_batch.batch_size
            if all_sparse:
                # all batch is sparse
                metadata = self.forward_metadata
                compressed_k = torch.full(
                    (forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, self.head_group_num, self.head_dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                    fill_value=float('-inf')
                )
                compressed_k2 = torch.full(
                    (forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, self.head_group_num, self.head_dim), 
                    dtype=torch.bfloat16, 
                    device=self.device,
                    fill_value=float('-inf')
                )

                get_compress_k_v2(
                    layer=layer,
                    forward_batch=forward_batch,
                    metadata=metadata,
                    full_compressed_k1=compressed_k, # output
                    full_compressed_k2=compressed_k2, # output
                    max_context_length=self.max_context_len,
                    page_size=self.page_size,
                )
                print("compressed_k")
                print(compressed_k)
                print(compressed_k2)
                
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_in_batch_k = metadata.max_seq_len_k
                cu_seqlens_q = metadata.cu_seqlens_q
                max_seqlen_in_batch_q = metadata.max_seq_len_q
                
                ret = self.sparse_get_topk_impl(
                            query_states,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            max_seqlen_in_batch_q,
                            max_seqlen_in_batch_k,
                            no_rope_param=no_rope_param,
                            compressed_k=compressed_k, compressed_cu_seqlens=metadata.k1.cu_seqlens,
                            compressed_k2=compressed_k2, compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                            fused_kernel=self.prefill_fused_kernels[forward_batch.batch_size] if self.fuse_topk else None
                )
                return ret

            topk_metadata = self.sparse_metadata_builder.build_prefill_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=self.forward_metadata,
                key_states=key_states,
                query_states=query_states,
                tp_q_head_num=layer.tp_q_head_num,
                head_dim=layer.head_dim,
                compress_k1_kernel_size=self.k1_kernel_size,
                compress_k1_kernel_stride=self.k1_kernel_stride,
                compress_k2_kernel_size=self.k2_kernel_size,
                compress_k2_kernel_stride=self.k2_kernel_stride,
                dense_len=self.dense_len,
            )

            sparse_bs = topk_metadata["sparse_bs"]
            topk_metadata["seqlens_q_sparse_bs"]
            topk_metadata["seqlens_k_sparse_bs"]
            k1_lens = topk_metadata["k1_lens"]
            k2_lens = topk_metadata["k2_lens"]

            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=sum(k1_lens),
                k2_token_nums=sum(k2_lens),
                dtype=key_states.dtype,
                device=key_states.device,
                max_context_length=self.max_context_len,
                split_stage1=self.split_stage1,
                page_size=self.page_size,
            )

            pt_k1, pt_k2 = 0, 0
            compressed_k = torch.zeros(
                (sum(k1_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            compressed_k2 = torch.zeros(
                (sum(k2_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )

            compressed_cu_seqlens, compressed_cu_seqlens2 = [0], [0]

            for sparse_bs_idx in sparse_bs:
                start = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx]
                end = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx + 1]
                compressed_k[pt_k1 : pt_k1 + (end - start), :, :] = full_compressed_k1[
                    start:end, :, :
                ]

                start2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx]
                end2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx + 1]
                compressed_k2[pt_k2 : pt_k2 + (end2 - start2), :, :] = (
                    full_compressed_k2[start2:end2, :, :]
                )

                pt_k1 += k1_lens[sparse_bs_idx]
                pt_k2 += k2_lens[sparse_bs_idx]
                compressed_cu_seqlens.append(
                    compressed_cu_seqlens[-1] + k1_lens[sparse_bs_idx]
                )
                compressed_cu_seqlens2.append(
                    compressed_cu_seqlens2[-1] + k2_lens[sparse_bs_idx]
                )

            compressed_cu_seqlens = torch.tensor(
                compressed_cu_seqlens, dtype=torch.int32, device=key_states.device
            )
            compressed_cu_seqlens2 = torch.tensor(
                compressed_cu_seqlens2, dtype=torch.int32, device=key_states.device
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            ret = self.sparse_get_topk_impl(
                query_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
                fused_kernel=self.prefill_fused_kernels[forward_batch.batch_size] if self.fuse_topk else None
            )
            return ret
        else:
            metadata = self.forward_metadata

            # if self.enable_cuda_graph:
            #     if self.split_stage1:
            get_compress_k_v2_padded(
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
                full_compressed_k1=self.graph_metadata["compress_k1"][:forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :],
                full_compressed_k2=self.graph_metadata["compress_k2"][:forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :],
                max_context_length=self.max_context_len,
                page_size=self.page_size,
            )
            #     else:
            #         get_compress_k_v2(
            #             layer=layer,
            #             forward_batch=forward_batch,
            #             metadata=metadata,
            #             full_compressed_k1=self.graph_metadata["compress_k1"][:forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :],
            #             full_compressed_k2=self.graph_metadata["compress_k2"][:forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :],
            #             max_context_length=self.max_context_len,
            #             page_size=self.page_size,
            #         )
            # else:
            #     compressed_k, compressed_k2 = allocate_and_compress_keys(
            #         layer=layer,
            #         forward_batch=forward_batch,
            #         metadata=metadata,
            #         k1_token_nums=forward_batch.batch_size
            #         * self.max_context_len
            #         // self.k1_kernel_stride,
            #         k2_token_nums=forward_batch.batch_size
            #         * self.max_context_len
            #         // self.k2_kernel_stride,
            #         dtype=torch.bfloat16,
            #         device=self.device,
            #         max_context_length=self.max_context_len,
            #         split_stage1=self.split_stage1,
            #         page_size=self.page_size,
            #     )

            topk_metadata = self.sparse_metadata_builder.build_decode_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                query_states=query_states,
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            # if self.enable_cuda_graph:
            ret = self.sparse_get_topk_impl(
                query_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=self.graph_metadata["compress_k1"][:forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :],
                compressed_cu_seqlens=metadata.k1.cu_seqlens,
                compressed_k2=self.graph_metadata["compress_k2"][:forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :],
                compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                fused_kernel=self.decode_fused_kernels[forward_batch.batch_size] if self.fuse_topk else None
            )

            # else:
                # ret = self.sparse_get_topk_impl(
                #     query_states,
                #     cu_seqlens_q,
                #     cu_seqlens_k,
                #     max_seqlen_in_batch_q,
                #     max_seqlen_in_batch_k,
                #     no_rope_param=no_rope_param,
                #     compressed_k=compressed_k,
                #     compressed_cu_seqlens=metadata.k1.cu_seqlens,
                #     compressed_k2=compressed_k2,
                #     compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                #     fused_kernel=self.decode_fused_kernels[forward_batch.batch_size] if self.fuse_topk else None
                # )

        return ret

    def sparse_get_topk_impl(
        self,
        query_layer,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        #    max_seqlen_k1,
        no_rope_param=None,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k2=None,
        compressed_cu_seqlens2=None,
        fused_kernel=None
    ):
        cache_lens = None
        if max_seqlen_in_batch_k > max_seqlen_in_batch_q:
            if max_seqlen_in_batch_q == 1:
                cache_lens = self.forward_metadata.cache_seqlens_int32_stage1
            else:
                seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                cache_lens = seq_lens_k - seq_lens_q
        else:
            batch_size = cu_seqlens_q.shape[0] - 1
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=cu_seqlens_q.device)

        max_seqlen_k = (max_seqlen_in_batch_k - self.kernel_size) // self.kernel_stride + 1 if max_seqlen_in_batch_k > self.kernel_size else 0
        return compressed_attention(
            (
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"]
            ),
            compressed_k,
            compressed_k2,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.sparse_topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            compressed_cu_seqlens2,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            self.max_context_len,
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            cu_seqlens_q_adjusted=self.forward_metadata.cu_seqlens_q_adjusted,
            max_seqlen_q_adjusted=self.forward_metadata.max_seqlen_q_adjusted,
            # block_score_buffer=self.forward_metadata.block_score_buffer
            split_stage1=self.split_stage1,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        # k1/k2 cache layout: [num_pages, page_size, num_kv_heads, head_size] -> 
        # [tokens, num_kv_heads, head_size] -> can be treated as non-paged
        max_k1_num_pages = (
            (self.max_context_len - self.k1_kernel_size) // self.k1_kernel_stride
            + 1
        )
        max_k2_num_pages = (
            (self.max_context_len - self.k2_kernel_size) // self.k2_kernel_stride
            + 1
        )
        sparse_max_num_pages = (
            self.num_sparse_topk_tokens + self.page_size - 1
        ) // self.page_size

        self.graph_metadata = {
            "page_table": torch.empty(
                (max_bs, max_num_pages),
                dtype=torch.int32,
                device=self.device,
            ),
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
            **(
                {
                    # sparse attention related metadata
                    # For sparse attention, cache_seqlens is fixed to num_sparse_topk_tokens
                    "sparse_cache_seqlens": torch.full(
                        (max_bs * 2,),
                        self.num_sparse_topk_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "sparse_cu_seqlens_q": torch.arange(
                        0, max_bs * 2 + 1, dtype=torch.int32, device=self.device
                    ),
                    # For sparse mode, cu_seqlens_k[i] = i * num_sparse_topk_tokens
                    "sparse_cu_seqlens_k": torch.arange(
                        0,
                        (max_bs * 2 + 1) * self.num_sparse_topk_tokens,
                        self.num_sparse_topk_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "token_to_bs": torch.arange(
                        0, max_bs, dtype=torch.int32, device=self.device
                    ),
                    "token_pos_in_bs": torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "sparse_page_table": torch.zeros(
                        max_bs * 2,
                        sparse_max_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    # TODO more precisely, it is max(0, (max_context_length - kernel_size) // kernel_stride + 1)
                    "compress_k1": torch.zeros(
                        (
                            max_bs * self.max_context_len // self.k1_kernel_stride,
                            self.head_group_num,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "compress_k2": torch.zeros(
                        (
                            max_bs * self.max_context_len // self.k2_kernel_stride,
                            self.head_group_num,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "k1.cu_seqlens": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_seqlens": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # too many arrays are stored, in order to support cuda graph, they are temporarily added,
                    # the parameters required for compress_k_core_new are reduced
                    # k1
                    "k1.table": torch.zeros(
                        max_bs,
                        max_k1_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "k1.history_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.new_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.new_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.total_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_new_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_new_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_total_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # k2
                    "k2.table": torch.zeros(
                        max_bs,
                        max_k2_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "k2.history_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.new_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.new_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.total_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_new_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_new_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_total_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # Stage1 optimization metadata
                    "cu_seqlens_q_adjusted": torch.arange(
                        0, max_bs + 1, dtype=torch.int32, device=self.device
                    ) * self.heads_per_group,
                    "cache_seqlens_int32_stage1": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                }
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        metadata = MiniCPMBackendMetadata()

        metadata.page_table = self.graph_metadata["page_table"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i * 1 for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
            batch_size = len(seq_lens)
            device = seq_lens.device
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.sparse_cache_seqlens_int32 = self.graph_metadata[
                "sparse_cache_seqlens"
            ][: batch_size * 2]
            metadata.sparse_cu_seqlens_q = self.graph_metadata[
                "sparse_cu_seqlens_q"
            ][: batch_size * 2 + 1]
            metadata.sparse_cu_seqlens_k = self.graph_metadata[
                "sparse_cu_seqlens_k"
            ][: batch_size * 2 + 1]
            metadata.token_to_bs = self.graph_metadata["token_to_bs"][
                :batch_size
            ]
            metadata.token_pos_in_bs = self.graph_metadata[
                "token_pos_in_bs"
            ][:batch_size]
            metadata.sparse_page_table = self.graph_metadata[
                "sparse_page_table"
            ][: batch_size * 2, :]

            metadata.k1 = CompressionLevelMetadata()
            metadata.k2 = CompressionLevelMetadata()

            metadata.k1.cu_seqlens = self.graph_metadata["k1.cu_seqlens"][
                : batch_size + 1
            ]
            metadata.k2.cu_seqlens = self.graph_metadata["k2.cu_seqlens"][
                : batch_size + 1
            ]
            assume_kv_len = self.config_dense_len
            assume_k1_len = (
                assume_kv_len - self.k1_kernel_size
            ) // self.k1_kernel_stride + 1
            assume_k2_len = (
                assume_kv_len - self.k2_kernel_size
            ) // self.k2_kernel_stride + 1
            for i in range(bs):
                metadata.cu_seqlens_k[i + 1] = metadata.cu_seqlens_k[i] + assume_kv_len
                metadata.k1.cu_seqlens[i + 1] = (
                    metadata.k1.cu_seqlens[i] + assume_k1_len
                )
                metadata.k2.cu_seqlens[i + 1] = (
                    metadata.k2.cu_seqlens[i] + assume_k2_len
                )

            metadata.max_seq_len_k = assume_kv_len
            metadata.k1.max_seq_len = assume_k1_len
            metadata.k2.max_seq_len = assume_k2_len

            # compress k1
            metadata.k1.history_compress_token_nums = self.graph_metadata[
                "k1.history_compress_token_nums"
            ][:batch_size]
            metadata.k1.new_token_nums = self.graph_metadata[
                "k1.new_token_nums"
            ][:batch_size]
            metadata.k1.new_compress_token_nums = self.graph_metadata[
                "k1.new_compress_token_nums"
            ][:batch_size]
            metadata.k1.total_compress_token_nums = self.graph_metadata[
                "k1.total_compress_token_nums"
            ][:batch_size]
            metadata.k1.cu_new_token_nums = self.graph_metadata[
                "k1.cu_new_token_nums"
            ][: batch_size + 1]
            metadata.k1.cu_new_compress_token_nums = self.graph_metadata[
                "k1.cu_new_compress_token_nums"
            ][: batch_size + 1]
            metadata.k1.cu_total_compress_token_nums = self.graph_metadata[
                "k1.cu_total_compress_token_nums"
            ][: batch_size + 1]
            # compress k2
            metadata.k2.history_compress_token_nums = self.graph_metadata[
                "k2.history_compress_token_nums"
            ][:batch_size]
            metadata.k2.new_token_nums = self.graph_metadata[
                "k2.new_token_nums"
            ][:batch_size]
            metadata.k2.new_compress_token_nums = self.graph_metadata[
                "k2.new_compress_token_nums"
            ][:batch_size]
            metadata.k2.total_compress_token_nums = self.graph_metadata[
                "k2.total_compress_token_nums"
            ][:batch_size]
            metadata.k2.cu_new_token_nums = self.graph_metadata[
                "k2.cu_new_token_nums"
            ][: batch_size + 1]
            metadata.k2.cu_new_compress_token_nums = self.graph_metadata[
                "k2.cu_new_compress_token_nums"
            ][: batch_size + 1]
            metadata.k2.cu_total_compress_token_nums = self.graph_metadata[
                "k2.cu_total_compress_token_nums"
            ][: batch_size + 1]

            metadata.k1.table = self.graph_metadata["k1.table"][:bs, :]
            metadata.k2.table = self.graph_metadata["k2.table"][:bs, :]

            # Stage1 optimization metadata
            metadata.cu_seqlens_q_adjusted = self.graph_metadata[
                "cu_seqlens_q_adjusted"
            ][: batch_size + 1]
            metadata.cache_seqlens_int32_stage1 = self.graph_metadata[
                "cache_seqlens_int32_stage1"
            ][:batch_size]
            # For decode mode, adjusted max_seqlen_q is fixed to heads_per_group
            metadata.max_seqlen_q_adjusted = metadata.max_seq_len_q * self.heads_per_group
        else:
            raise NotImplementedError(
                "MiniCPM backend CUDA graph capture only supports decode/idle mode, "
                f"got {forward_mode}"
            )
            
        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        metadata = self.graph_metadata[bs]
        seq_lens = seq_lens[:bs]
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.page_table[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.page_table[:bs, max_seq_pages:].fill_(0)
        metadata.page_table[bs:, :].fill_(0)
        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])
        metadata.max_seq_len_k = max_len
        metadata.cache_seqlens_int32.copy_(seq_lens[:bs])
        metadata.cu_seqlens_k[1:].copy_(torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32))
        
        real_bs = forward_batch.sparse_cache_seqlens_int32_cpu.numel() // 2

        metadata.sparse_cache_seqlens_int32[: 2 * real_bs].copy_(
            forward_batch.sparse_cache_seqlens_int32_cpu
        )
        metadata.sparse_cu_seqlens_k[: 2 * real_bs + 1].copy_(
            forward_batch.sparse_cu_seqlens_k_cpu
        )

        # Stage1 optimization metadata update
        metadata.cache_seqlens_int32_stage1[:real_bs].copy_(
            forward_batch.cache_seqlens_int32_stage1_cpu
        )

        self.graph_metadata["compress_k1"][:forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :].fill_(float('-inf')) 
        self.graph_metadata["compress_k2"][:forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :].fill_(float('-inf'))
        metadata.k1.cu_seqlens[: real_bs + 1].copy_(forward_batch.cu_seqlens_k1_cpu)
        metadata.k2.cu_seqlens[: real_bs + 1].copy_(forward_batch.cu_seqlens_k2_cpu)

        metadata.k1.history_compress_token_nums[:real_bs].copy_(
            forward_batch.history_compress_k1_token_nums_cpu
        )
        metadata.k2.history_compress_token_nums[:real_bs].copy_(
            forward_batch.history_compress_k2_token_nums_cpu
        )

        metadata.k1.new_token_nums[:real_bs].copy_(
            forward_batch.new_k1_token_nums_cpu
        )
        metadata.k2.new_token_nums[:real_bs].copy_(
            forward_batch.new_k2_token_nums_cpu
        )
        metadata.k1.cu_new_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_new_k1_token_nums_cpu
        )
        metadata.k2.cu_new_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_new_k2_token_nums_cpu
        )

        metadata.k1.new_compress_token_nums[:real_bs].copy_(
            forward_batch.new_compress_k1_token_nums_cpu
        )
        metadata.k2.new_compress_token_nums[:real_bs].copy_(
            forward_batch.new_compress_k2_token_nums_cpu
        )
        metadata.k1.cu_new_compress_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_new_compress_k1_token_nums_cpu
        )
        metadata.k2.cu_new_compress_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_new_compress_k2_token_nums_cpu
        )

        metadata.k1.total_compress_token_nums[:real_bs].copy_(
            forward_batch.total_compress_k1_token_nums_cpu
        )
        metadata.k2.total_compress_token_nums[:real_bs].copy_(
            forward_batch.total_compress_k2_token_nums_cpu
        )
        metadata.k1.cu_total_compress_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_total_compress_k1_token_nums_cpu
        )
        metadata.k2.cu_total_compress_token_nums[: real_bs + 1].copy_(
            forward_batch.cu_total_compress_k2_token_nums_cpu
        )

        if real_bs < bs:
            metadata.sparse_cache_seqlens_int32[2 * real_bs : ].fill_(0)
            metadata.sparse_cu_seqlens_k[2 * real_bs + 1 : ].fill_(forward_batch.sparse_cu_seqlens_k_cpu[-1])
            metadata.cache_seqlens_int32_stage1[real_bs:].fill_(0)
            metadata.k1.cu_seqlens[real_bs + 1 :].fill_(forward_batch.cu_seqlens_k1_cpu[-1])
            metadata.k2.cu_seqlens[real_bs + 1 :].fill_(forward_batch.cu_seqlens_k2_cpu[-1])
            metadata.k1.history_compress_token_nums[real_bs:].fill_(0)
            metadata.k2.history_compress_token_nums[real_bs:].fill_(0)
            metadata.k1.new_token_nums[real_bs:].fill_(0)
            metadata.k2.new_token_nums[real_bs:].fill_(0)
            metadata.k1.cu_new_token_nums[real_bs + 1 :].fill_(forward_batch.cu_new_k1_token_nums_cpu[-1])
            metadata.k2.cu_new_token_nums[real_bs + 1 :].fill_(forward_batch.cu_new_k2_token_nums_cpu[-1])
            metadata.k1.new_compress_token_nums[real_bs:].fill_(0)
            metadata.k2.new_compress_token_nums[real_bs:].fill_(0)

            metadata.k1.cu_new_compress_token_nums[real_bs + 1 :].fill_(forward_batch.cu_new_compress_k1_token_nums_cpu[-1])
            metadata.k2.cu_new_compress_token_nums[real_bs + 1 :].fill_(forward_batch.cu_new_compress_k2_token_nums_cpu[-1])
            metadata.k1.total_compress_token_nums[real_bs:].fill_(0)
            metadata.k2.total_compress_token_nums[real_bs:].fill_(0)

            metadata.k1.cu_total_compress_token_nums[real_bs + 1 :].fill_(forward_batch.cu_total_compress_k1_token_nums_cpu[-1])

            metadata.k2.cu_total_compress_token_nums[real_bs + 1 :].fill_(forward_batch.cu_total_compress_k2_token_nums_cpu[-1])
            
        metadata.k1.table.copy_(self.req_to_sparse_k1_token[req_pool_indices])
        metadata.k2.table.copy_(self.req_to_sparse_k2_token[req_pool_indices])
                
        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            raise NotImplementedError(
                "MiniCPM backend does not support extend for target verify or draft extend"
            )

        torch.set_printoptions(
            edgeitems=3,      # 每行前后显示的元素数，其他显示 ... 省略
            threshold=1000,   # 总元素数大于 threshold 时触发省略
            linewidth=80,     # 每行字符宽度
            precision=4,      # 小数位数
            profile="default" # 保留默认打印风格
        )
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
            
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Get the appropriate page table (without local attention or SWA support)
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens_k = metadata.cu_seqlens_k
        
        bs = forward_batch.batch_size
        if max(forward_batch.seq_lens_cpu) >= self.dense_len:
            q_reshaped = q.contiguous().view(
                -1, layer.tp_q_head_num, layer.head_dim
            )
            topk_idx = self.get_topk_for_sparse(
                q_reshaped, k, v, q.shape[0], layer, forward_batch
            )
            print("topk_idx")
            print(topk_idx)
            
            sparse_page_table_sparse_bs = get_sparse_block_table(
                topk_idx,
                page_table,
                metadata.token_to_bs,
                metadata.token_pos_in_bs,
                metadata.seqlen_k_sparse_bs_tensor,
                self.sparse_topk,
                self.page_size,
                self.block_size,
            ).reshape(-1, self.num_sparse_topk_blocks)
            print("page_table")
            print(page_table)
            print(sparse_page_table_sparse_bs.shape)
            # torch.set_printoptions(threshold=torch.inf)
            print(sparse_page_table_sparse_bs)
            # torch.set_printoptions(
            #     edgeitems=3,      # 每行前后显示的元素数，其他显示 ... 省略
            #     threshold=1000,   # 总元素数大于 threshold 时触发省略
            #     linewidth=80,     # 每行字符宽度
            #     precision=4,      # 小数位数
            #     profile="default" # 保留默认打印风格
            # )
            # copy page table for sparse bs
            print(forward_batch.sparse_idx)
            metadata.sparse_page_table[forward_batch.sparse_idx, :self.num_sparse_topk_blocks] = sparse_page_table_sparse_bs
        else:
            total_k1 = self.forward_metadata.k1.cu_total_compress_token_nums[-1].item()
            total_k2 = self.forward_metadata.k2.cu_total_compress_token_nums[-1].item()

            full_compressed_k1_ext, full_compressed_k2_ext = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                dtype=k.dtype,
                device=k.device,
                max_context_length=self.max_context_len,
                split_stage1=self.split_stage1,
                page_size=self.page_size,
            )

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim)
        if forward_batch.sparse_batch_size < bs:
            # copy dense page table for dense bs
            metadata.sparse_page_table.shape[1]
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                kv_len = forward_batch.seq_lens_cpu[dense_bs]
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert (
                    sparse_page_table_idx_end - sparse_page_table_idx_start == 2
                ), "dense bs should have 2 head_group, but get {}".format(
                    sparse_page_table_idx_end - sparse_page_table_idx_start
                )

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert (
                    len_ == forward_batch.extend_seq_lens_cpu[dense_bs]
                ), "dense bs seqlen mismatch {} vs {}".format(
                    len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                )
                t = q_reshaped[ps : ps + 2 * len_, :, :].clone()
                q_reshaped[ps : ps + len_, :, :] = t[0::2, :, :]
                q_reshaped[ps + len_ : ps + 2 * len_, :, :] = t[1::2, :, :]

                metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = page_table[dense_bs, : kv_len] * 2
                metadata.sparse_page_table[sparse_page_table_idx_start + 1, : kv_len] = page_table[dense_bs, : kv_len] * 2 + 1
        
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.head_dim
        )
        v_cache = v_cache.view(
            -1, self.page_size, layer.head_dim
        )
        
        num_tokens = q.shape[0]
        
        num_heads_per_group = layer.tp_q_head_num // layer.tp_k_head_num
        
        attn_output = torch.empty(
            (num_tokens * layer.tp_k_head_num, 1, num_heads_per_group * layer.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        
        t = q.contiguous().view(-1, 1, layer.tp_q_head_num // layer.tp_k_head_num, layer.head_dim)
        print("prefill attn inputs")
        print(metadata.sparse_page_table.shape)
        print(forward_batch.sparse_cache_seqlens_int32_cpu)
        print(metadata.sparse_page_table)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            t,
            k_cache,
            v_cache,
            num_heads=layer.tp_q_head_num // layer.tp_k_head_num,
            num_key_value_heads=1,
            input_layout="BSND",
            atten_mask=None,
            block_size=self.page_size,
            block_table=metadata.sparse_page_table,
            actual_seq_lengths_kv=forward_batch.sparse_cache_seqlens_int32_cpu,
            scale=layer.scaling,
        )
        # workspace = (
        #     torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        #         t,
        #         k_cache,
        #         v_cache,
        #         block_table=metadata.sparse_page_table,
        #         block_size=self.page_size,
        #         num_heads=num_heads_per_group,
        #         num_key_value_heads=1,
        #         input_layout="BSH",
        #         scale=layer.scaling,
        #         actual_seq_lengths_kv=forward_batch.sparse_cache_seqlens_int32_cpu,
        #     )
        # )
        
        # softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
        # torch_npu.npu_fused_infer_attention_score.out(
        #     t,
        #     k_cache,
        #     v_cache,
        #     block_table=metadata.sparse_page_table,
        #     block_size=self.page_size,
        #     num_heads=num_heads_per_group,
        #     num_key_value_heads=1,
        #     input_layout="BSH",
        #     scale=layer.scaling,
        #     actual_seq_lengths_kv=forward_batch.sparse_cache_seqlens_int32_cpu,
        #     workspace=workspace,
        #     out=[attn_output, softmax_lse]
        # )
        

        # torch_npu._npu_paged_attention(
        #     query=q.contiguous().view(-1, layer.tp_q_head_num // layer.tp_k_head_num, layer.head_dim),
        #     key_cache=k_cache,
        #     value_cache=v_cache,
        #     num_heads=layer.tp_q_head_num // layer.tp_k_head_num,
        #     num_kv_heads=1,
        #     scale_value=layer.scaling,
        #     block_table=metadata.sparse_page_table,
        #     context_lens=forward_batch.sparse_cache_seqlens_int32_cpu,
        #     out=attn_output,
        # )

        if forward_batch.sparse_batch_size < bs:
            metadata.sparse_page_table.shape[1]
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert (
                    sparse_page_table_idx_end - sparse_page_table_idx_start == 2
                ), "dense bs should have 2 head_group"

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert (
                    len_ == forward_batch.extend_seq_lens_cpu[dense_bs]
                ), "dense bs seqlen mismatch {} vs {}".format(
                    len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                )
                t = attn_output[ps : ps + 2 * len_, :, :].clone()
                attn_output[ps : ps + 2 * len_ : 2, :, :] = t[0:len_, :, :]
                attn_output[ps + 1 : ps + 2 * len_ : 2, :, :] = t[len_ : 2 * len_, :, :]
                
        res = attn_output.view(
            num_tokens, layer.tp_q_head_num * layer.v_head_dim
        )

        print("prefill res")
        print(res)        
        return attn_output.view(
            num_tokens, layer.tp_q_head_num * layer.v_head_dim
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        bs = forward_batch.batch_size
        metadata = self.forward_metadata

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        topk_idx = self.get_topk_for_sparse(
            q_reshaped.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            1,
            layer,
            forward_batch,
            False,
        )
        
        # print("topk_idx")
        # print(topk_idx)

        sparse_page_table = get_sparse_block_table(
            topk_idx,
            page_table,
            metadata.token_to_bs,
            cache_seqlens,
            cache_seqlens,
            self.sparse_topk,
            self.page_size,
            self.block_size,
        ).reshape(-1, self.num_sparse_topk_blocks)
        
        # print("sparse_page_table")
        # print(sparse_page_table)
            
        metadata.sparse_page_table[: 2 * bs, : self.num_sparse_topk_blocks] = (
            sparse_page_table[:, : self.num_sparse_topk_blocks]
        )

        q_reshaped_by_head_group = q_reshaped.reshape(
            -1, layer.tp_q_head_num // 2, layer.head_dim
        )
        key_cache_by_head_group = key_cache.reshape(
            -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
        )
        value_cache_by_head_group = value_cache.reshape(
            -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
        )
          
        # prepare seqlen_k and it's presum
        sparse_cache_seqlens = metadata.sparse_cache_seqlens_int32
        sparse_cu_seqlens_k = metadata.sparse_cu_seqlens_k
        sparse_cu_seqlens_q = metadata.sparse_cu_seqlens_q

        attn_output = torch.empty(
            (bs * 2, layer.tp_q_head_num // 2, layer.head_dim),
            dtype=q.dtype,
            device=q.device,
        )

        torch_npu._npu_paged_attention(
            query=q_reshaped_by_head_group,
            key_cache=key_cache_by_head_group,
            value_cache=value_cache_by_head_group,
            num_heads=layer.tp_q_head_num // 2,
            num_kv_heads=layer.tp_k_head_num // 2,
            scale_value=layer.scaling,
            block_table=metadata.sparse_page_table,
            context_lens=forward_batch.sparse_cache_seqlens_int32_cpu,
            out=attn_output,
        )
        
        # print("decode res")
        # print(attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim))
        
        return attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)