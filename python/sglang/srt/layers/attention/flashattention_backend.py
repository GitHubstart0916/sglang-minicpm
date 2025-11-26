from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

import torch.nn.functional as F
from einops import rearrange, repeat
from functools import lru_cache

from infllm_v2 import (
    infllmv2_attn_stage1,
    infllmv2_attn_varlen_func,
    infllmv2_attn_with_kvcache,
    max_pooling_1d,
    max_pooling_1d_varlen
)

@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq]

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed

# hard code fa code, due to version diff
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None
    
index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply

def unpad_input(hidden_states, attention_mask, unused_mask=None):
        """
        Arguments:
            hidden_states: (batch, seqlen, ...)
            attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
            unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
        Return:
            hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
            indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
            cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
            max_seqlen_in_batch: int
            seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
        """
        all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
        seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
        used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
        # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
        # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
        # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
        # so we write custom forward and backward to make it a bit faster.
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            used_seqlens_in_batch, 
        )
        
def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Module for compressing key (K) representations.

        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: torch.Tensor, cu_seqlens):
        """
        Forward pass for compressing the key (K) tensor.

        Args:
            k (torch.Tensor): Input key tensor of shape (total_seq_len, num_heads, head_dim).
            cu_seqlens (torch.Tensor): Cumulative sequence lengths for each sample in the batch, typically used for handling variable-length sequences.

        Returns:
            compress_k (torch.Tensor): Compressed key tensor.
            cu_seqlens_compressed (torch.Tensor): Updated cumulative sequence lengths after compression.

        """
        # Compute chunk-related metadata, with stride support
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        # Extract filtered key vectors
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))

        # split
        filtered_k = filtered_k.view(filtered_k.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim)  # [l, block_size,h,d]

        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed
    

def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        
        # Check if it's prefilling stage
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        
        if is_prefilling:  # prefilling stage
            # Calculate q_idx for each query position in each batch
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device) 
            q_idx = torch.cat([
                (torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device) + 
                 max_seqlen_q - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])) // block_size
                for i in range(batch_size)
            ], dim=0)  # shape: [total_q_len]
        else:  # decoding stage
            # Each batch has only one query (last position)
            q_idx = cache_lens // block_size  # shape: [batch_size] = [total_q_len] in decoding

        # 计算attention score
        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_prefilling
        )
        score = score[:, :q_idx.shape[0], :]  # [num_heads, total_q_len, num_blocks]
        
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride
        )  # shape: [num_heads, total_q_len, num_blocks]
        

        # get topk
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

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
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


# Copied from:
# https://github.com/houseroad/vllm/blob/4e45bfcaf928bdb9bd952b4ac922a3c205589ae8/vllm/v1/attention/backends/flash_attn.py
#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
    local attention blocks, where each block is passed to the attention kernel
    as an independent local ("virtual") batch item.

    Args:
        attn_chunk_size: Size of local attention chunks
        query_start_loc_np: Cumulative sum of query lengths (numpy array)
        seq_lens_np: Sequence lengths (numpy array)
        block_table: Block table for KV cache
        page_size: Size of each page in the KV cache

    Returns:
        seqlens_q_local: Query sequence lengths for local attention
        cu_seqlens_q_local: Cumulative sum of query sequence lengths for local attention
        seqlens_k_local: Key sequence lengths for local attention
        block_table_local: Block table for local attention
    """
    # Adjust attention_chunk_size based on the actual sequence length
    # to avoid index out of bounds errors
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    # Make sure effective_chunk_size is divisible by page_size
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming page_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    # Ensure block_indices doesn't exceed block_table dimensions
    # This is a critical safety check that prevents index out of bounds errors
    # when dealing with large sequences (>8192 tokens) or when the block_table
    # dimensions are smaller than what would be needed for the full attention chunk size.
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# TODO(hebiao064): remove this once we have a better way to handle the merge_state_v2 torch.compile issue
@torch._dynamo.disable()
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )
        
        
        # compress args
        self.kernel_size = 32
        self.kernel_stride = 16
        self.init_blocks = 1
        self.block_size = 64
        self.window_size = 2048
        self.dense_len = 8192

        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.topk = 64 + (self.window_size // self.block_size)
        self.use_nope = False
        
        self.compress_k1_len = 0
        self.compress_k2_len = 0
        
        
        
        self.compress_k = CompressK(2, 128, kernel_size=self.kernel_size, kernel_stride=self.kernel_stride)
        self.compress_k2 = CompressK(2, 128, kernel_size=self.kernel_size*4, kernel_stride=self.kernel_stride*4)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.max_seq_len_k = self.speculative_step_id + 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    cache_loc = forward_batch.out_cache_loc.view(
                        self.speculative_num_steps, -1
                    ).T.contiguous()
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata_expand.max_seq_len_k = (
                    metadata_expand.cache_seqlens_int32.max().item()
                )
                self.forward_metadata_spec_decode_expand = metadata_expand
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata


    def sparse_attn_forward(
        self, 
        query_states, 
        key_states, 
        value_states, 
        query_length,  
        layer,
        forward_batch,
        is_prefill=True,
        test_prefill=False,
        dropout=0.0, 
        softmax_scale=None, 
        no_rope_param=None, 
        past_key_value=None,
    ):
        
        
        # print("sparse_attn_forward: q shape: {}, k shape : {}, v shape: {}".format(query_states.shape, key_states.shape, value_states.shape))
        
        if is_prefill:
            
            bs, seqlens_q, seqlens_k, k1_lens, k2_lens = forward_batch.batch_size, forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens_cpu, forward_batch.token_num_sparse_16_cpu, forward_batch.token_num_sparse_64_cpu
            pt, pt_k1, pt_k2 = 0, 0, 0
            compressed_k = torch.zeros((sum(forward_batch.token_num_sparse_16_cpu), layer.tp_k_head_num, layer.head_dim), dtype=key_states.dtype, device=key_states.device)
            compressed_cu_seqlens = torch.zeros((bs + 1,), dtype=torch.int32, device=key_states.device)
            compressed_k2 = torch.zeros((sum(forward_batch.token_num_sparse_64_cpu), layer.tp_k_head_num, layer.head_dim), dtype=key_states.dtype, device=key_states.device)
            compressed_cu_seqlens2 = torch.zeros((bs + 1,), dtype=torch.int32, device=key_states.device)
            for i in range(bs):
                
                attention_mask = torch.ones(1, seqlens_q[i], dtype=torch.int64, device=query_states.device)
                        
                # write compressed kv cache, read all past compressed k cache
                compressed_k[pt_k1 : pt_k1 + k1_lens[i], :, :], k1, compressed_k2[pt_k2 : pt_k2 + k2_lens[i], :, :], k2 = self._get_compress_k(
                    key_states=key_states.unsqueeze(0)[:, pt : pt + seqlens_q[i], : , :],
                    attention_mask=attention_mask,
                    layer=layer,
                    forward_batch=forward_batch,
                    batch_id=i
                )
                assert k1[1] == k1_lens[i], "k1 shape mismatch, expected {}, got {}".format(k1_lens[i], k1[1])
                assert k2[1] == k2_lens[i], "k2 shape mismatch, expected {}, got {}".format(k2_lens[i], k2[1])
                pt += seqlens_q[i]
                pt_k1 += k1_lens[i]
                pt_k2 += k2_lens[i]
                compressed_cu_seqlens[i + 1] = compressed_cu_seqlens[i] + k1_lens[i]
                compressed_cu_seqlens2[i + 1] = compressed_cu_seqlens2[i] + k2_lens[i]
            
            # if layer.layer_id == 0:
            #     compressed_k.cpu().view(torch.uint16).numpy().tofile("prefill_compressed_k_{}_{}.bin".format(compressed_k.shape[0], layer.layer_id))
            #     compressed_k2.cpu().view(torch.uint16).numpy().tofile("prefill_compressed_k2_{}_{}.bin".format(compressed_k2.shape[0], layer.layer_id))
            
            
            # if layer.layer_id == 0:
            #     print("query_states shape: {}, key_states shape: {}, value_states shape: {}".format(query_states.shape, key_states.shape, value_states.shape))
    
            
            # query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            #             query_states, key_states, value_states, attention_mask, query_length
            #         )
            cu_seqlens_q = torch.cumsum(torch.tensor([0] + seqlens_q, dtype=torch.int32, device=query_states.device), dim=0, dtype=torch.int32)
            cu_seqlens_k = torch.cumsum(torch.tensor([0] + seqlens_k, dtype=torch.int32, device=query_states.device), dim=0, dtype=torch.int32)
            max_seqlen_in_batch_q = max(seqlens_q)
            max_seqlen_in_batch_k = max(seqlens_k)
            
            query_states = query_states.reshape(-1, layer.tp_q_head_num, layer.head_dim)
            key_states = key_states.reshape(-1, layer.tp_k_head_num, layer.head_dim)
            value_states = value_states.reshape(-1, layer.tp_k_head_num, layer.head_dim)
            
            # if layer.layer_id == 0:
            #     print("sparse_get_topk_impl query_states shape: {}, key_states shape: {}, value_states shape: {}".format(query_states.shape, key_states.shape, value_states.shape))
            #     print("sparse_get_topk_impl cu_seqlens_q: {}, cu_seqlens_k: {}, max_seqlen_in_batch_q: {}, max_seqlen_in_batch_k: {}".format(cu_seqlens_q, cu_seqlens_k, max_seqlen_in_batch_q, max_seqlen_in_batch_k))
            #     print("sparse_get_topk_impl compressed_k shape: {}, compressed_cu_seqlens: {}, compressed_k2 shape: {}, compressed_cu_seqlens2: {}".format(compressed_k.shape, compressed_cu_seqlens, compressed_k2.shape, compressed_cu_seqlens2))
                
            
            
            
            # cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens  
            if test_prefill:
                ret = self.sparse_get_topk_impl(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_in_batch_q,
                        max_seqlen_in_batch_k,
                        no_rope_param=no_rope_param,
                        compressed_k=compressed_k, compressed_cu_seqlens=compressed_cu_seqlens,
                        compressed_k2=compressed_k2, compressed_cu_seqlens2=compressed_cu_seqlens2
                    )
                return ret
            attn_output_unpad = self.sparse_forward_impl(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_in_batch_q,
                        max_seqlen_in_batch_k,
                        no_rope_param=no_rope_param,
                        compressed_k=compressed_k, compressed_cu_seqlens=compressed_cu_seqlens,
                        compressed_k2=compressed_k2, compressed_cu_seqlens2=compressed_cu_seqlens2
                    )

            ret = pad_input(attn_output_unpad, indices_q, 1, query_length)
        else:
            # bs = forward_batch.batch_size
            
            # assert query_states.shape[0] == bs, "Speculative Decoding is not yet supported."
            
            # cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=query_states.device)
            # cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=query_states.device)
            # max_seqlen_in_batch_k = 0
            # max_seqlen_in_batch_q = 1
            
            # for i in range(bs):
            #     kv_len = forward_batch.seq_lens_cpu[i]
            #     attention_mask = torch.ones(1, kv_len, dtype=torch.int64, device=query_states.device)
            #     compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2 = self._get_compress_k(
            #         key_states=key_states[i:i+1, :, :],
            #         attention_mask=attention_mask,
            #         layer=layer,
            #         forward_batch=forward_batch,
            #         batch_id=i
            #     )
            #     # query_states, key_states, value_states, _, _, _ = self._upad_input(
            #     #     query_states, key_states, value_states, attention_mask, query_length
            #     # )
            #     # create cu_seqlens
            #     cu_seqlens_k[i + 1] = cu_seqlens_k[i] + kv_len
            #     max_seqlen_in_batch_k = max(kv_len, max_seqlen_in_batch_k) if i > 0 else kv_len
            # query_states = query_states.reshape(-1, query_states.shape[2], query_states.shape[3])
            
                
            bs = query_states.shape[0]
            assert bs == 1
            kv_len = forward_batch.seq_lens_cpu[0]
            attention_mask = torch.ones(bs, kv_len, dtype=torch.int64, device=query_states.device)
            compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2 = self._get_compress_k(
                key_states=key_states,
                attention_mask=attention_mask,
                layer=layer,
                forward_batch=forward_batch
            )
            # query_states, key_states, value_states, _, _, _ = self._upad_input(
            #     query_states, key_states, value_states, attention_mask, query_length
            # )
            query_states = query_states.reshape(-1, query_states.shape[2], query_states.shape[3])
            # as bs = 1, we can directly create cu_seqlens
            cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=query_states.device)
            cu_seqlens_k[1] = kv_len
            max_seqlen_in_batch_k = kv_len
            cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=query_states.device)
            cu_seqlens_q[1] = query_states.shape[0]
            max_seqlen_in_batch_q = query_states.shape[0]
            
                          
            ret = self.sparse_get_topk_impl(
                        query_states,
                        key_states, 
                        value_states,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_in_batch_q,
                        max_seqlen_in_batch_k,
                        no_rope_param=no_rope_param,
                        compressed_k=compressed_k, compressed_cu_seqlens=compressed_cu_seqlens,
                        compressed_k2=compressed_k2, compressed_cu_seqlens2=compressed_cu_seqlens2
                    )

        return ret
    
    def sparse_forward_impl(self,
                       query_layer,
                       key_layer,
                       value_layer,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       max_seqlen_in_batch_q,
                       max_seqlen_in_batch_k,
                       no_rope_param=None,
                       compressed_k=None, compressed_cu_seqlens=None,
                       compressed_k2=None, compressed_cu_seqlens2=None):
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        cache_lens = None
        if max_seqlen_in_batch_q==1 and max_seqlen_in_batch_k>1: #decoding
            seq_lens_k =  cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k-1
        

        topk_idx = compressed_attention(
            query_layer if no_rope_param is None else no_rope_param['query_states_no_rope'],
            compressed_k,
            compressed_k2,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            compressed_cu_seqlens2,
            max_seqlen_in_batch_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens
        )
            
        topk_attn_output = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=max_seqlen_in_batch_q != 1,
            return_attn_probs=False,
            # block_window_size=self.window_size // self.block_size,
            topk_idx=topk_idx
        )

        return topk_attn_output

    
    def sparse_get_topk_impl(self,
                       query_layer,
                       key_layer,
                       value_layer,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       max_seqlen_in_batch_q,
                       max_seqlen_in_batch_k,
                       no_rope_param=None,
                       compressed_k=None, compressed_cu_seqlens=None,
                       compressed_k2=None, compressed_cu_seqlens2=None):
        
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        cache_lens = None
        if max_seqlen_in_batch_q==1 and max_seqlen_in_batch_k>1: #decoding
            seq_lens_k =  cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k-1

        # print("sparse_get_topk_impl max_seqlen_in_batch_q {}, max_seqlen_in_batch_k {}, cache_lens {}".format(
        #     max_seqlen_in_batch_q, max_seqlen_in_batch_k, cache_lens
        # ))
        # print("Topk args 1 sparse_get_topk_impl max_seqlen_in_batch_q {}, max_seqlen_in_batch_k {}, cache_lens {}".format(
        #         max_seqlen_in_batch_q, max_seqlen_in_batch_k, cache_lens
        #     ))
        # print("Topk args 2 query_layer shape {}, key_layer shape {}, value_layer shape {}".format(
        #     query_layer.shape, key_layer.shape, value_layer.shape
        # ))
        # print("Topk args 3 cu_seqlens_q {}, cu_seqlens_k {}".format(
        #     cu_seqlens_q, cu_seqlens_k
        # ))
        # print("Topk args 4 compressed_cu_seqlens {}, compressed_cu_seqlens2 {}".format(
        #     compressed_cu_seqlens, compressed_cu_seqlens2
        # ))
        # print("Topk args 5 compressed_k shape {}, compressed_k2 shape {}".format(
        #     compressed_k.shape, compressed_k2.shape
        # ))
        topk_idx = compressed_attention(
            query_layer if no_rope_param is None else no_rope_param['query_states_no_rope'],
            compressed_k,
            compressed_k2,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            compressed_cu_seqlens2,
            max_seqlen_in_batch_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens
        )

        return topk_idx
    
    def _get_compress_k(self, key_states, attention_mask, 
                        layer,
                        forward_batch,
                        batch_id = 0):
        # only support batch size 1 for now
        req_id = forward_batch.req_pool_indices[batch_id]
        past_compress_k1_token_num = forward_batch.req_to_token_pool.compress_k1_len[req_id]
        past_compress_k2_token_num = forward_batch.req_to_token_pool.compress_k2_len[req_id]
        
        k1_compress_idx_st = 0 if past_compress_k1_token_num == 0 else (past_compress_k1_token_num - 1) * 16 + 16
        k2_compress_idx_st = 0 if past_compress_k2_token_num == 0 else (past_compress_k2_token_num - 1) * 64 + 64
        
        token_num = forward_batch.seq_lens_cpu[batch_id].item()
        # print("Req id {}, past compress k1 token num {}, past compress k2 token num {}, current token num {}, k1_compress_idx_st {}, k2_compress_idx_st {}".format(
        #     req_id, past_compress_k1_token_num, past_compress_k2_token_num, token_num, k1_compress_idx_st, k2_compress_idx_st
        # ))
        
        key_cache,_ = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
        )
        
        key_cache = key_cache.view(
                -1, layer.tp_k_head_num, layer.head_dim
        )[forward_batch.req_to_token_pool.req_to_token[req_id][:token_num]].unsqueeze(0)
        
        k1 = key_cache[:, k1_compress_idx_st:token_num, :, :]
        k2 = key_cache[:, k2_compress_idx_st:token_num, :, :]
        
        # print("k1 shape {}, k2 shape {}".format(k1.shape, k2.shape))


        new_k1_num, new_k2_num = 0, 0
        
        if k1.shape[1] >= 32:
            attention_mask = torch.ones(k1.shape[0], k1.shape[1], dtype=torch.int64, device=k1.device)
            unpadded_key_states, _, cu_seqlens, _ = self._unpad_one_tensor(k1,attention_mask=attention_mask)
            compressed_k1, compressed_cu_seqlens = self.compress_k(unpadded_key_states, cu_seqlens)
            seq_len = k1.shape[1]
            res_len = (seq_len - ((seq_len - 32) // 16 * 16 + 32)) + 16
            new_k1_num = (seq_len - 32) // 16 + 1
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.sparse_16_loc[
                    torch.sum(forward_batch.token_num_sparse_16_cpu[:batch_id]) 
                    : torch.sum(forward_batch.token_num_sparse_16_cpu[:batch_id + 1])], 
                compressed_k1, compressed_k1, None, None
            )
            # print("Compress k1 from length {} to length {}, write loc len is {}, res len is {}".format(seq_len, compressed_k1.shape[0], forward_batch.sparse_16_loc.shape, res_len))

        if k2.shape[1] >= 128:
            attention_mask = torch.ones(k2.shape[0], k2.shape[1], dtype=torch.int64, device=k2.device)
            unpadded_key_states, _, cu_seqlens, _ = self._unpad_one_tensor(k2,attention_mask=attention_mask)
            compressed_k2, compressed_cu_seqlens2 = self.compress_k2(unpadded_key_states, cu_seqlens)
            seq_len = k2.shape[1]
            res_len2 = (seq_len - ((seq_len - 128) // 64 * 64 + 128)) + 64
            new_k2_num = (seq_len - 128) // 64 + 1
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.sparse_64_loc[torch.sum(forward_batch.token_num_sparse_64_cpu[:batch_id]) 
                    : torch.sum(forward_batch.token_num_sparse_64_cpu[:batch_id + 1])], 
                compressed_k2, compressed_k2, None, None
            )
            # print("Compress k2 from length {} to length {}, write loc len is {}, res len is {}".format(seq_len, compressed_k2.shape[0], forward_batch.sparse_64_loc.shape, res_len2))
        
        
        compressed_k1, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
        )
        compressed_k1 = compressed_k1[forward_batch.req_to_token_pool.req_to_sparse_16_token[forward_batch.req_pool_indices[batch_id]][:forward_batch.req_to_token_pool.compress_k1_len[req_id] + new_k1_num]]
        compressed_cu_seqlens = torch.tensor([0, compressed_k1.shape[0]], device=compressed_k1.device, dtype=torch.int32)

        compressed_k2, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
        )
        compressed_k2 = compressed_k2[forward_batch.req_to_token_pool.req_to_sparse_64_token[forward_batch.req_pool_indices[batch_id]][:forward_batch.req_to_token_pool.compress_k2_len[req_id] + new_k2_num]]
        compressed_cu_seqlens2 = torch.tensor([0, compressed_k2.shape[0]], device=compressed_k2.device, dtype=torch.int32)
        
        # print("compressed_k shape {}, compressed_k2 shape {}".format(compressed_k1.shape, compressed_k2.shape))
        # print("cum_seqlens {}, compressed_cu_seqlens2 {}".format(compressed_cu_seqlens, compressed_cu_seqlens2))
        if forward_batch.seq_lens_cpu[batch_id] == 8208 and layer.layer_id == 0:
            compressed_k1.cpu().view(torch.uint16).numpy().tofile("debug_decode_compressed_k1_{}_{}.bin".format(compressed_k1.shape[0], layer.layer_id))
            compressed_k2.cpu().view(torch.uint16).numpy().tofile("debug_decode_compressed_k2_{}_{}.bin".format(compressed_k2.shape[0], layer.layer_id))
            # print("debug_decode_compressed_k1_ compressed_k1[-1] {} {}".format(
            #     compressed_k1[-1][0], compressed_k1[-1][0]
            # ))
    
        return compressed_k1, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2
    
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = self._get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, 32, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = self.unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    
    def _unpad_one_tensor(self, hidden_states, attention_mask):
        # Unpad the hidden states using the indices
        indices, cu_seqlens, max_seqlen_in_batch = self._get_unpad_data(attention_mask)
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Get the remaining dimensions
        remaining_dims = hidden_states.shape[2:]
        
        # Reshape to (batch_size * seq_len, *remaining_dims)
        reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
        
        # Apply unpadding using indices
        unpadded_states = index_first_axis(reshaped_states, indices)
        
        return unpadded_states, indices, cu_seqlens, max_seqlen_in_batch
    
    def _get_unpad_data(self, attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None
        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
            k_descale = layer.k_scale.expand(descale_shape)
            v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify() and self.topk > 1
        )

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            max_seqlen_k = metadata.max_seq_len_k
            cu_seqlens_k = metadata.cu_seqlens_k
        
        # if layer.layer_id == 0:
        #     print("self.use_mla {}, use_local_attn {}, use_cascade_attn {}, k_descale {}, v_descale {}, kv_cache_type {}, memory_saver_adapter {}".
        #           format(self.use_mla, use_local_attn, 
        #                  use_cascade_attn, k_descale, v_descale,
        #                  type(forward_batch.token_to_kv_pool).__name__,
        #                  forward_batch.token_to_kv_pool.memory_saver_adapter))
        #     print("page_table ", page_table)
        #     print("page size ", self.page_size)
        #     key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
        #         layer.layer_id
        #     )
        #     key_cache = key_cache.view(
        #         -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        #     )
        #     print("first page tensor", key_cache.shape, key_cache[page_table[0][0]], key_cache[page_table[0][0]].shape)
            
        if q.shape[0] >= 8192:
            # split batch here
            bs, seqlens_q = forward_batch.batch_size, forward_batch.extend_seq_lens_cpu
            # attn_output = self.sparse_attn_forward(q.unsqueeze(0), 
            #                                     k.unsqueeze(0), 
            #                                     v.unsqueeze(0), 
            #                                     q.shape[0], 
            #                                     layer, 
            #                                     forward_batch)
            # attn_output = attn_output.reshape(q.shape[0], 4096)
            # # print(attn_output)
            # attn_output.cpu().view(torch.uint16).numpy().tofile("attn_output_{}_{}.bin".format(q.shape[0], layer.layer_id))
            
            # q_rashaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            # if layer.layer_id == 0:
            #     print("start get topk idx, q shape {}, k shape {}, v shape {}, bs {}, seqlens_q {}".format(
            #         q.shape, k.shape, v.shape, bs, seqlens_q
            #     ))
            topk_idx = self.sparse_attn_forward(q, 
                                                k, 
                                                v, 
                                                q.shape[0], 
                                                layer, 
                                                forward_batch,
                                                test_prefill=True)
            # pt = 0
            # for i in range(bs):
            #     topk_idx = self.sparse_attn_forward(q[pt : pt + seqlens_q[i]].unsqueeze(0), 
            #                                         k[pt : pt + seqlens_q[i]].unsqueeze(0), 
            #                                         v[pt : pt + seqlens_q[i]].unsqueeze(0), 
            #                                         seqlens_q[i], 
            #                                         layer, 
            #                                         forward_batch,
            #                                         test_prefill=True)
            #     pt += seqlens_q[i]
            #     print("topk_idx shape {} device {}".format(topk_idx.shape, topk_idx.device))
            import sparse_kernel_extension
            
            q_shape_0 = topk_idx.shape[1]
            seqlen_q_as_param = (metadata.cu_seqlens_q.diff())
            meta_cu_seqlens_q = metadata.cu_seqlens_q
            bs = metadata.cu_seqlens_q.shape[0] - 1
            assert q_shape_0 == seqlen_q_as_param.sum().item(), "q_shape_0 {} vs seqlen_q_as_param sum {}".format(q_shape_0, seqlen_q_as_param.sum().item())
            
            # print("batch_size is {}".format(bs))
            token_to_bs = torch.zeros(q_shape_0, dtype=torch.int32, device=topk_idx.device)
            for i in range(bs):
                token_to_bs[meta_cu_seqlens_q[i] : meta_cu_seqlens_q[i + 1]] = i
            
            # token_pos_in_bs = torch.tensor([_ for _ in range(1, q_shape_0 + 1)], dtype=torch.int32, device=topk_idx.device)
            
            token_pos_in_bs = torch.zeros(q_shape_0, dtype=torch.int32, device=topk_idx.device)
            for i in range(bs):
                token_pos_in_bs[meta_cu_seqlens_q[i] : meta_cu_seqlens_q[i + 1]] = torch.tensor(
                    [(idx + 1) for idx in range(seqlen_q_as_param[i].item())], dtype=token_to_bs.dtype, device=token_to_bs.device)
            
            # if layer.layer_id == 0:
            #     print("start get block table, param shape is topk_idx {}, page_table {}, token_to_bs {}, token_pos_in_bs {}, cu_seqlens_q diff shape {}".format(
            #         topk_idx.shape, page_table.shape, token_to_bs.shape, token_pos_in_bs.shape, seqlen_q_as_param.shape
            #     ))
            #     print("start get block table, param device is topk_idx {}, page_table {}, token_to_bs {}, token_pos_in_bs {}, cu_seqlens_q diff shape {}".format(
            #         topk_idx.device, page_table.device, token_to_bs.device, token_pos_in_bs.device, seqlen_q_as_param.device
            #     ))
            #     print("start get block table, param dtype is topk_idx {}, page_table {}, token_to_bs {}, token_pos_in_bs {}, cu_seqlens_q diff shape {}".format(
            #         topk_idx.dtype, page_table.dtype, token_to_bs.dtype, token_pos_in_bs.dtype, seqlen_q_as_param.dtype
            #     ))
            #     print("start get block table, check value {}".format(seqlen_q_as_param))
            
            # if layer.layer_id == 0:
            #     page_table.cpu().numpy().tofile("extend_dense_page_table_{}.bin".format(layer.layer_id))
            #     topk_idx.cpu().numpy().tofile("extend_topk_idx_{}.bin".format(layer.layer_id))
            
            sparse_page_table = sparse_kernel_extension.get_block_table(
                topk_idx,
                page_table,
                token_to_bs,
                token_pos_in_bs,
                seqlen_q_as_param
            )
            
            # if layer.layer_id == 0:
                
            #     sparse_page_table.cpu().numpy().tofile("extend_sparse_page_table_{}.bin".format(layer.layer_id))
            
            # print("output page_table shape {}, item: {} {}".format(sparse_page_table.shape, (sparse_page_table[8191][0] != 0).sum(), (sparse_page_table[8191][1] != 0).sum()))
            # update sparse metadata
            # page_table
            # cu_seqlens_q 
            # cache_seqlens
            # max_seqlen_q 
            # max_seqlen_k 
            # cu_seqlens_k 
            # batched_token_num = topk_idx.shape[1]
            # head_group_num = topk_idx.shape[0]
            # sparse_bs = topk_idx.shape[0] * topk_idx.shape[1]
            
            # sparse_cu_seqlens_q_cpu = [0 for _ in range(sparse_bs + 1)]
            # sparse_cache_lens_cpu = [0 for _ in range(sparse_bs)]
            
            # sparse_cu_seqlens_q = torch.zeros(sparse_bs + 1, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
            # sparse_cache_lens = torch.zeros(sparse_bs, dtype=cache_seqlens.dtype, device=cache_seqlens.device)
            
            
            # print("start prepare token->bs_idx table")
            # token_pos_to_bs = [0 for _ in range(cu_seqlens_q[-1])]
            # for bs in range(forward_batch.seq_lens_cpu.shape[0]):
            #     for pos in range(forward_batch.seq_lens_cpu[bs]):
            #         token_pos_to_bs[pos + cu_seqlens_q[bs]] = bs
            # print("end prepare token->bs_idx table, table shape is {}".format(len(token_pos_to_bs)))
            
            # print("start prepare sparse page table")
            # sparse_topk = topk_idx.shape[2]
            
            
            # topk_idx_cpu = topk_idx.cpu().numpy()
            # cu_seqlens_q_cpu = cu_seqlens_q.cpu().numpy()
            # page_table_cpu = page_table.cpu().numpy()
            
            # sparse_page_table_cpu = np.zeros((batched_token_num * head_group_num, sparse_topk * 64), dtype=page_table_cpu.dtype)
            
            # temp = np.arange(batched_token_num * head_group_num)
            
            # for i in range(batched_token_num):
            #     # sparse_cu_seqlens_q[i+1] = sparse_cu_seqlens_q[i] + batched_token_num
            #     bs = token_pos_to_bs[i]
            #     if layer.layer_id == 0 and i % 1000 == 0:
            #         print("process token idx {}, bs {}, cu_seqlens_q {}".format(i, bs, cu_seqlens_q))
            #     for head_group in range(head_group_num):
            #         # sparse_page_table_cpu.append([])
            #         sparse_cu_seqlens_q_cpu[i * head_group_num + head_group + 1] = sparse_cu_seqlens_q_cpu[i * head_group_num + head_group] + 1
            #         cache_len = 0
            #         # TODO: Fix me for batch size > 1
            #         max_k_idx = i - cu_seqlens_q_cpu[bs] + 1
                    
            #         for j in range(sparse_topk):
            #             block_idx = int(topk_idx_cpu[head_group][i][j].item())
            #             if block_idx >= 0:
            #                 # [2 * page_table_cpu[bs][id] + head_group for id in range(block_idx * 64, min(max_k_idx, block_idx * 64 + 64))]
            #                 # token_idx_arrange = np.arange(block_idx * 64, min(max_k_idx, block_idx * 64 + 64))
            #                 token_idx_arrange = temp[block_idx * 64: min(max_k_idx, block_idx * 64 + 64)]
                            
            #                 ext = 2 * page_table_cpu[bs][token_idx_arrange] + head_group
            #                 # print(ext.shape)
            #                 # ext = np.array([2 * page_table_cpu[bs][id] + head_group for id in range(block_idx * 64, min(max_k_idx, block_idx * 64 + 64))])
            #                 sparse_page_table_cpu[i * head_group_num + head_group][j * 64 : j * 64 + ext.shape[0]] = ext
            #                 cache_len += 64
                    
            #         sparse_cache_lens_cpu[i * head_group_num + head_group] = cache_len
            
            # print("end prepare sparse page table")
            
            # # construct tensors
            # sparse_page_table = torch.tensor(sparse_page_table_cpu, dtype=page_table.dtype, device=page_table.device)
            
            sparse_cache_lens = torch.tensor([(sparse_page_table[i][0] != 0).sum() for i in range(topk_idx.shape[1])], dtype=cache_seqlens.dtype, device=cache_seqlens.device).repeat_interleave(2)
            sparse_cu_seqlens_q = torch.tensor([i for i in range(topk_idx.shape[1] * 2 + 1)], dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
            sparse_max_seqlen_q = 1 # since we treat prefill as multi-batch decode
            sparse_max_seqlen_k = 6144 # du to page_table shape
            sparse_cu_seqlens_k = torch.cat([torch.zeros(1, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device),
                                                torch.cumsum(sparse_cache_lens, dim=0, dtype=cu_seqlens_k.dtype)],
                                                dim=0)
            
            # if layer.layer_id == 0:
            #     print("dense param: page_table {}, cu_seqlens_q {}, cache_seqlens {}, max_seqlen_q {}, max_seqlen_k {}, cu_seqlens_k {}".format(
            #         page_table.shape, cu_seqlens_q, cache_seqlens, max_seqlen_q, max_seqlen_k, cu_seqlens_k
            #     ))
            #     print("sparse param: page_table {}, cu_seqlens_q {}, cache_seqlens {}, max_seqlen_q {}, max_seqlen_k {}, cu_seqlens_k {}".format(
            #         sparse_page_table.shape, sparse_cu_seqlens_q, sparse_cache_lens, sparse_max_seqlen_q, sparse_max_seqlen_k, sparse_cu_seqlens_k
                # ))
                # exit(0)
                # save page_table & topk for eval
                
                # exit(0)
                
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
            )
            
            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=sparse_page_table.reshape(-1, 6144),
                cache_seqlens=sparse_cache_lens,
                cu_seqlens_q=sparse_cu_seqlens_q,
                cu_seqlens_k_new=sparse_cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=sparse_max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
            )
            # print("forward extend call flash-attn twice, {}".format(result.shape))
            result = result.reshape(q.shape[0], 4096)
            # print("forward extend call flash-attn twice, {}".format(result.shape))
            # print(result[-1])
        
            # if layer.layer_id == 0 and q.shape[0] == 16384:
            #     result.cpu().view(torch.uint16).numpy().tofile("attn_output_new_{}_{}.bin".format(q.shape[0], layer.layer_id))
            
            attn_output = result.reshape(q.shape[0], 4096)
            return attn_output

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            # result = flash_attn_with_kvcache(
            #     q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            #     k_cache=key_cache,
            #     v_cache=value_cache,
            #     page_table=page_table,
            #     cache_seqlens=cache_seqlens,
            #     cu_seqlens_q=cu_seqlens_q,
            #     cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
            #     max_seqlen_q=max_seqlen_q,
            #     softmax_scale=layer.scaling,
            #     causal=False if use_cascade_attn else causal,
            #     window_size=window_size,
            #     softcap=layer.logit_cap,
            #     k_descale=k_descale,
            #     v_descale=v_descale,
            #     return_softmax_lse=use_cascade_attn,
            # )
            # print("forward extend call flash-attn twice")
            result_1 = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)[:, 0:16, :],
                k_cache=key_cache[:, :, 0:1, :],
                v_cache=value_cache[:, :, 0:1, :],
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
            )
            
            result_2 = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)[:, 16:32, :],
                k_cache=key_cache[:, :, 1:2, :],
                v_cache=value_cache[:, :, 1:2, :],
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
            )
            
            result = torch.cat([result_1, result_2], dim=1)
            # if layer.layer_id == 0:
            #     print("result.shape {} result_1.shape {}, result_2.shape {}".format(result.shape, result_1.shape, result_2.shape))

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                not global_server_args_dict["disable_chunked_prefix_cache"]
                and forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache

                if forward_batch.attn_attend_prefix_cache:
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    output, lse, *rest = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output, lse, *rest = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=True,
                    )
                return output, lse
            else:
                # Do absorbed multi-latent attention
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result


        # if layer.layer_id == 0 and q.shape[0] == 8192:
        #     attn_output = self.sparse_attn_forward(q.reshape(1, 8192, 32, 128), 
        #                                         k.reshape(1, 8192, 2, 128), 
        #                                         v.reshape(1, 8192, 2, 128), 8192, 
        #                                         layer, forward_batch)
        #     attn_output = attn_output.reshape(8192, 4096)
        #     print("in fa backend, attn_output is {}".format(attn_output.shape))
        #     # print(attn_output)
        #     attn_output.cpu().view(torch.uint16).numpy().tofile("attn_output_fa.bin")
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        bs = forward_batch.batch_size
        # if bs != 1:
        #     return torch.zeros((bs, 4096), dtype=q.dtype, device=q.device)
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None
        if self.kv_cache_dtype_str != "auto":
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)

        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )
                # if layer.layer_id == 0:
                #     print("q_reshaped shape ", q_reshaped.shape)
                    
                page_table1, page_table2 = page_table, page_table
                
                # torch suggest that, to create a tensor with data without compute graph, use clone().detach()
                # in sparse topk, only last block can not full, but due to algorithm, last block will always be selected,
                # so, for each head_group in same token, sparse cache len is same 
                # sparse_cu_seqlen_k = cu_seqlens_k.clone().detach()
                sparse_cache_seqlens = cache_seqlens.clone().detach()
                 
                page_table_cpu = []
                for head_group in range(layer.tp_k_head_num):
                    page_table_cpu.append([])
                
                for b in range(bs):
                    if forward_batch.seq_lens_cpu[b] >= 8192:
                        topk_idx = self.sparse_attn_forward(q_reshaped[b:b+1, :, :].unsqueeze(0), 
                                                    k[b:b+1, :, :].unsqueeze(0), 
                                                    v[b:b+1, :, :].unsqueeze(0), 
                                                    1, 
                                                    layer, 
                                                    forward_batch,
                                                    False)
                        # if layer.layer_id == 0:
                        #     print("topk_idx shape {} page_table shape {}".format(topk_idx.shape, page_table.shape))
                        #     topk_idx.cpu().numpy().tofile("bs_{}_topk_idx_{}_{}_{}.bin".format(bs, forward_batch.seq_lens_cpu[b], layer.layer_id, b))
                        
                        # TODO: change this to modern python code 
                           
                        assert topk_idx.shape[1] == 1, "topk_idx shape[1] {} vs 1".format(topk_idx.shape[1])
                        for head_group in range(topk_idx.shape[0]):
                            for i in range(topk_idx.shape[1]):
                                max_k_idx = forward_batch.seq_lens_cpu[b]
                                page_table_cpu[head_group].append([])
                                for j in range(topk_idx.shape[2]):
                                    block_idx = int(topk_idx[head_group][i][j].item())
                                    page_table_cpu[head_group][b].extend([page_table[b][id] for id in range(block_idx * 64, min(max_k_idx, block_idx * 64 + 64))])
                        
                        sparse_cache_seqlens[b] = len(page_table_cpu[0][b])
                    else:
                        for head_group in range(layer.tp_k_head_num):
                            page_table_cpu[head_group].append(page_table[b])
                        
                        sparse_cache_seqlens[b] = cache_seqlens[b] 

                tensors = [torch.tensor(d, dtype=page_table.dtype) for d in page_table_cpu[0]]
                page_table1 = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0).to(device=page_table.device)   
                tensors = [torch.tensor(d, dtype=page_table.dtype) for d in page_table_cpu[1]]   
                page_table2 = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0).to(device=page_table.device)   
                            
                # page_table1 = torch.tensor(page_table_cpu[0], device=page_table.device, dtype=page_table.dtype)
                # page_table2 = torch.tensor(page_table_cpu[1], device=page_table.device, dtype=page_table.dtype)
                
                # if layer.layer_id == 0:
                #     print("after sparse attention, page_table1 shape {}, page_table2 shape {}, page_table shape {}".
                #         format(page_table1.shape, page_table2.shape, page_table.shape))
                #     print("first 10 elements {} {} {}".format(page_table1, page_table2, page_table))
                #     print("cu_seqlens_k {} cache_seqlens {}".format(cu_seqlens_k, cache_seqlens))
                
                
                
                # cu_seqlens_k[1], cache_seqlens[0] = page_table1.shape[1], page_table1.shape[1]
                for i in range(bs):
                    cache_seqlens[i] = sparse_cache_seqlens[i]
                    cu_seqlens_k[i + 1] = cu_seqlens_k[i] + cache_seqlens[i]
            
                
            
                # if layer.layer_id == 0:   
                #     print("after update cu_seqlens_k {} cache_seqlens {}".format(cu_seqlens_k, cache_seqlens))        
                    
                      
                    
                
                # Default: single-token self-attention
                # result = flash_attn_with_kvcache(
                #     q=q_reshaped,
                #     k_cache=key_cache,
                #     v_cache=value_cache,
                #     page_table=page_table,
                #     cache_seqlens=cache_seqlens,
                #     cu_seqlens_q=metadata.cu_seqlens_q,
                #     cu_seqlens_k_new=cu_seqlens_k,
                #     max_seqlen_q=max_seqlen_q,
                #     softmax_scale=layer.scaling,
                #     causal=False if use_cascade_attn else causal,
                #     window_size=window_size,
                #     softcap=layer.logit_cap,
                #     k_descale=k_descale,
                #     v_descale=v_descale,
                #     return_softmax_lse=use_cascade_attn,
                # )
                # if forward_batch.seq_lens_cpu[0] >= 8192:
                
                q_reshaped_by_head_group = q_reshaped.reshape(-1, layer.tp_q_head_num // 2, layer.head_dim)
                assert self.page_size == 1
                key_cache_by_head_group = key_cache.reshape(-1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim)
                value_cache_by_head_group = value_cache.reshape(-1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim)
                
                # page_table1.shape [token_num, ]
                # this should be pre sum of all seq, this write is mock, due to only support batch size 1
                # TODO: fix for batch size > 1, cu_seqlens_k and cache_seqlens need to be recomputed [0, 1, 2, 3, ..., 2 * batch_size], [1, 1, 1, ....]
                # prepare seqlen_k presum
                # cu_seqlens_k_cat = torch.tensor([0, page_table1.shape[1], page_table1.shape[1] * 2], device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
                # cache_seqlens_cat = torch.tensor([page_table1.shape[1], page_table2.shape[1]], device=cache_seqlens.device, dtype=cache_seqlens.dtype)
                
                # prepare seqlen_k and it's presum
                cache_seqlens_cat = sparse_cache_seqlens.repeat_interleave(2)
                cu_seqlens_k_cat = torch.cat([torch.zeros(1, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device),
                                                torch.cumsum(cache_seqlens_cat, dim=0, dtype=cu_seqlens_k.dtype)],
                                                dim=0)
                # prepare seqlen_q presum
                seqlens_q_bk = torch.diff(metadata.cu_seqlens_q)
                seqlens_q_cat = seqlens_q_bk.repeat_interleave(2)
                cu_seqlens_q_cat = torch.cat([torch.zeros(1, dtype=metadata.cu_seqlens_q.dtype, device=metadata.cu_seqlens_q.device), 
                                                torch.cumsum(seqlens_q_cat, dim=0, dtype=metadata.cu_seqlens_q.dtype)], 
                                                dim=0)
                # cu_seqlens_q_cat = torch.tensor([0, 1, 2], device=metadata.cu_seqlens_q.device, dtype=metadata.cu_seqlens_q.dtype)
                # page_table_cat = torch.cat([page_table1 * 2, page_table2 * 2 + 1], dim=0)
                page_table_cat = torch.zeros((page_table1.shape[0] * 2, page_table1.shape[1]), dtype=page_table.dtype, device=page_table.device)
                page_table_cat[0::2] = page_table1 * 2
                page_table_cat[1::2] = page_table2 * 2 + 1
                
                # if layer.layer_id == 0:
                #     print("check tensor shape, q_reshaped_by_head_group {}, key_cache_by_head_group {}, value_cache_by_head_group {}, page_table_cat {},"
                #           " not cat shape is q_reshaped {}, k_cache {}, v_cache {}, page_table1 {}".
                #           format(q_reshaped_by_head_group.shape, key_cache_by_head_group.shape, value_cache_by_head_group.shape, page_table_cat.shape,
                #                  q_reshaped[:, 0:16, :].shape, key_cache[:, :, 0:1, :].shape, value_cache[:, :, 0:1, :].shape, page_table1.shape))
                    
                #     print("after update cu_seqlens_k_cat {} cache_seqlens_cat {}".format(cu_seqlens_k_cat, cache_seqlens_cat))
                #     print("after update metadata.cu_seqlens_q is {}, cu_seqlens_q_cat is {}".format(metadata.cu_seqlens_q, cu_seqlens_q_cat))
                    


                result_cat = flash_attn_with_kvcache(
                    q=q_reshaped_by_head_group,
                    k_cache=key_cache_by_head_group,
                    v_cache=value_cache_by_head_group,
                    page_table=page_table_cat,
                    cache_seqlens=cache_seqlens_cat,
                    cu_seqlens_q=cu_seqlens_q_cat,
                    cu_seqlens_k_new=cu_seqlens_k_cat,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                
                # result1 = flash_attn_with_kvcache(
                #     q=q_reshaped[:, 0:16, :],
                #     k_cache=key_cache[:, :, 0:1, :],
                #     v_cache=value_cache[:, :, 0:1, :],
                #     page_table=page_table1,
                #     cache_seqlens=cache_seqlens,
                #     cu_seqlens_q=metadata.cu_seqlens_q,
                #     cu_seqlens_k_new=cu_seqlens_k,
                #     max_seqlen_q=max_seqlen_q,
                #     softmax_scale=layer.scaling,
                #     causal=False if use_cascade_attn else causal,
                #     window_size=window_size,
                #     softcap=layer.logit_cap,
                #     k_descale=k_descale,
                #     v_descale=v_descale,
                #     return_softmax_lse=use_cascade_attn,
                # )
                
                # result_2 = flash_attn_with_kvcache(
                #     q=q_reshaped[:, 16:32, :],
                #     k_cache=key_cache[:, :, 1:2, :],
                #     v_cache=value_cache[:, :, 1:2, :],
                #     page_table=page_table2,
                #     cache_seqlens=cache_seqlens,
                #     cu_seqlens_q=metadata.cu_seqlens_q,
                #     cu_seqlens_k_new=cu_seqlens_k,
                #     max_seqlen_q=max_seqlen_q,
                #     softmax_scale=layer.scaling,
                #     causal=False if use_cascade_attn else causal,
                #     window_size=window_size,
                #     softcap=layer.logit_cap,
                #     k_descale=k_descale,
                #     v_descale=v_descale,
                #     return_softmax_lse=use_cascade_attn,
                # )
                
                # result = torch.cat([result1, result_2], dim=1)
                result = result_cat
                
                # if layer.layer_id == 0 or layer.layer_id == 1:
                #     result.cpu().view(torch.uint16).numpy().tofile("q_len_{}_kv_len_{}_fa_result_{}.bin".format(q.shape[0], page_table1.shape[1], layer.layer_id))
                #     result_cat.cpu().view(torch.uint16).numpy().tofile("q_len_{}_kv_len_{}_cat_result_{}.bin".format(q.shape[0], page_table1.shape[1], layer.layer_id))
                
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_table_draft_decode": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        # Only allocate local attention buffers if local attention is enabled
        # This prevents OOM errors when local attention is not being used
        if self.attention_chunk_size is not None:
            # Estimate maximum sizes for local attention metadata
            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        # This is used by draft decode's first half of metadata when topk > 1
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # This is used by draft decode's second half of metadata when topk > 1
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    (self.max_context_len + self.page_size - 1) // self.page_size,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    (self.max_context_len + self.page_size - 1) // self.page_size,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        self.encoder_metadata = {
            "encoder_page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "encoder_lens_int32": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "encoder_cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
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
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()

        # metadata_expand is needed for Spec Decoding when top k > 1
        metadata_expand = FlashAttentionMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.max_seq_len_k = seq_lens.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][req_pool_indices, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = seq_lens.max().item()
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][req_pool_indices, :]

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.max_seq_len_k = (
                        self.speculative_step_id + 1
                    )  # , do this in replay
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    req_pool_indices, :
                ]
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                self.decode_cuda_graph_metadata[bs] = metadata

                if self.attention_chunk_size is not None:
                    self._update_local_attn_metadata_for_capture(metadata, batch_size)

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
                )

                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    seq_lens.max().item() + self.speculative_num_draft_tokens
                )

                metadata.cu_seqlens_q = torch.arange(
                    0,
                    bs * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )

                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]

                metadata.page_table = self.target_verify_metadata["page_table"][
                    req_pool_indices, :
                ]

                self.target_verify_metadata[bs] = metadata
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                # metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item(), do this in replay
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][req_pool_indices, :]

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]

                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand
        elif forward_mode.is_draft_extend():
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))

            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            metadata.page_table = self.draft_extend_metadata["page_table"][
                req_pool_indices, :
            ]

            self.draft_extend_metadata[bs] = metadata

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]

            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                req_pool_indices, :
            ]

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    metadata = self.decode_cuda_graph_metadata[bs]
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32.copy_(
                        (seq_lens + (self.speculative_step_id + 1)).to(torch.int32)
                    )

                    metadata.max_seq_len_k = seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_k[1:].copy_(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        )
                    )

                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size
                    page_indices = self.req_to_token[
                        req_pool_indices[:, None],
                        self.decode_cuda_graph_metadata["strided_indices"][
                            :max_seq_pages
                        ],
                    ]

                    page_indices //= self.page_size
                    metadata.page_table[:, :max_seq_pages].copy_(page_indices)
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
                    # metadata.max_seq_len_q = self.topk, already set in capture
                    metadata.max_seq_len_k = seq_lens_cpu.max().item()
                    # metadata.cu_seqlens_q already set in capture
                    metadata.cu_seqlens_k[1:].copy_(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        )
                    )

                    page_table = self.req_to_token[
                        req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    cache_loc = out_cache_loc.view(
                        self.speculative_num_steps, -1
                    ).T.contiguous()
                    metadata_expand.page_table[: cache_loc.shape[0]].copy_(
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                # TODO: Handle local attention metadata for draft decode when llama4 eagle is supported
            else:
                metadata = self.decode_cuda_graph_metadata[bs]
                # Normal Decode
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                # Optimize cumulative sequence length calculation
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
                )

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][
                        None, :
                    ],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
                metadata.page_table[:, max_seq_pages:].fill_(0)

                self._update_local_attn_metadata_for_replay(metadata, bs)
        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
                )

                metadata.max_seq_len_k = (
                    seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
                # metadata.max_seq_len_q = self.speculative_num_draft_tokens, already set in capture
                metadata.max_seq_len_k = seq_lens_cpu.max().item()
                # metadata.cu_seqlens_q already set in capture
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                page_table = self.req_to_token[
                    req_pool_indices, : metadata.max_seq_len_k
                ]
                metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand = self.target_verify_metadata_topk_expand[bs]
                # metadata_expand.max_seq_len_q = 1, already set in capture
                # metadata_expand.cu_seqlens_q already set in capture

                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # avoid extracting padded seq indices which will be out of boundary
                mask_extraction_indices[
                    :, spec_info.positions.numel() * self.speculative_num_draft_tokens :
                ].fill_(0)

                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(
                    mask.sum(dim=1).to(torch.int32)
                )
                metadata_expand.cu_seqlens_k[1:].copy_(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32,
                        dim=0,
                        dtype=torch.int32,
                    )
                )
                metadata_expand.max_seq_len_k = (
                    metadata_expand.cache_seqlens_int32.max().item()
                )
        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            accept_length = spec_info.accept_length[:bs]
            metadata.max_seq_len_q = accept_length.max().item()
            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            page_indices //= self.page_size
            metadata.page_table[:, :max_seq_pages].copy_(page_indices)

        if encoder_lens is not None:
            # Only support encoder size 1 for now
            metadata.encoder_max_seq_len_k = encoder_lens[0]
            metadata.encoder_lens_int32.copy_(encoder_lens[:1])
            metadata.encoder_cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32)
            )

            metadata.encoder_page_table[:, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )

            # Update the regular page table
            page_table = self.req_to_token[
                req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]
            metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0

    def _init_local_attn_metadata(self, metadata: FlashAttentionMetadata, device):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update local attention metadata during CUDA graph capture phase.

        This method calculates the exact buffer sizes needed for local attention metadata
        during the CUDA graph capture phase, optimizing memory usage by creating views of
        pre-allocated buffers with exactly the sizes needed.
        """
        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        cu_seqlens_q_np = metadata.cu_seqlens_q.cpu().numpy()
        seqlens_np = seq_lens_capture.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            page_table_capture,
            self.page_size,
        )

        # Get exact dimensions from the calculation
        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1

        # Create views of the pre-allocated buffers with exactly these sizes
        # This is the key optimization - we only use the memory we actually need
        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def _update_local_attn_metadata_for_replay(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update preallocated local attention metadata in-place before CUDA graph replay."""
        if self.attention_chunk_size is None:
            return

        # Access preallocated buffers
        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]

        # Create a modified version for local attention that only processes the last token
        # This mimics the normal decode pattern
        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        # Slice the page_table to match the batch size and actual sequence length
        # This serves three important purposes:
        # 1. Ensures we only process the actual batch size (bs) and not the maximum batch size
        # 2. Limits the sequence length to prevent processing padding tokens or garbage values
        # 3. Prevents zeros in the block table which can cause garbage output during replay
        #
        # Without this slicing, the pre-allocated page_table may contain zeros or invalid indices
        # beyond the actual sequence length, leading to incorrect attention calculations
        max_seq_len = int(seqlens.max().item())
        sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seqlens_np = seqlens.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            sliced_page_table,
            self.page_size,
        )

        # Convert back to tensors
        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)
        # Get sizes
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        # In-place updates into preallocated tensors and zero out the unused space
        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())


class FlashAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        print("init fa backend, speculative_num_steps is {}".format(speculative_num_steps))
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )
