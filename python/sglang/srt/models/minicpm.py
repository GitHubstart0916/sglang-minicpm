# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniCPM model compatible with HuggingFace weights."""

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

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
 
class MiniCPMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        # compress args
        self.kernel_size = 32
        self.kernel_stride = 16
        self.init_blocks = 1
        self.block_size = 64
        self.window_size = 2048
        self.dense_len = 8192

        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.topk = 64 + (self.window_size//self.block_size)
        self.use_nope = False
        
        self.compress_k1_len = 0
        self.compress_k2_len = 0
        
        
        self.compress_k = CompressK(self.total_num_kv_heads, self.head_dim, kernel_size=self.kernel_size, kernel_stride=self.kernel_stride)
        self.compress_k2 = CompressK(self.total_num_kv_heads, self.head_dim, kernel_size=self.kernel_size*4, kernel_stride=self.kernel_stride*4)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # assert forward_batch.batch_size == 1, "Only batch size 1 is supported in MiniCPM for now."
        if self.layer_id == 0:
            print("Minicpm layer id {}, forward batch, batch_size {} seq_lens {}".format(self.layer_id, forward_batch.batch_size, forward_batch.seq_lens))
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        orig_dtype = q.dtype
        q, k = q.float(), k.float()
        q, k = self.rotary_emb(positions, q, k)
        q, k = q.to(orig_dtype), k.to(orig_dtype)
        # if self.layer_id == 0 and q.shape[0] == 8192:
        #     print(q.shape, k.shape, v.shape)
        #     print(q, k, v)
        # if self.layer_id == 0:
        #     print("Minicpm batch {}, seq_len {}, cache_loc {}, sparse_16_loc {}, sparse_64_loc {}".
        #             format(forward_batch.batch_size, forward_batch.seq_lens, forward_batch.out_cache_loc, forward_batch.sparse_16_loc, forward_batch.sparse_64_loc))
        #     print("Minicpm page_table {}, sparse_16_page_table {}, sparse_64_page_table {}".
        #             format(forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[0]][:forward_batch.seq_lens[0]],
        #                    forward_batch.req_to_token_pool.req_to_sparse_16_token[forward_batch.req_pool_indices[0]][:self.compress_k1_len],
        #                    forward_batch.req_to_token_pool.req_to_sparse_64_token[forward_batch.req_pool_indices[0]][:self.compress_k2_len]))
            
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        
        # if self.layer_id == 0 and q.shape[0] >= 8192:
        #     attn_output_cpm = self._sparse_attn_forward(q.reshape(1, q.shape[0], 32, 128), k.reshape(1, q.shape[0], 2, 128), v.reshape(1, q.shape[0], 2, 128), q.shape[0])
        #     attn_output_cpm = attn_output_cpm.reshape(q.shape[0], 4096)
        #     print(attn_output_cpm.shape)
        #     attn_output_cpm.cpu().view(torch.uint16).numpy().tofile("attn_output_cpm_{}.bin".format(q.shape[0]))
            
        if self.layer_id == 31:
            if forward_batch.sparse_16_loc is not None:
                forward_batch.req_to_token_pool.compress_k1_len[forward_batch.req_pool_indices[0]] += len(forward_batch.sparse_16_loc)
            if forward_batch.sparse_64_loc is not None:
                forward_batch.req_to_token_pool.compress_k2_len[forward_batch.req_pool_indices[0]] += len(forward_batch.sparse_64_loc)
        
        return output
    
    def _sparse_attn_forward(
            self, 
            query_states, 
            key_states, 
            value_states, 
            query_length, 
            dropout=0.0, 
            softmax_scale=None, 
            no_rope_param=None, 
            past_key_value=None
        ):
        # only support use_nope == False state
        # attention_mask is needed
        # q, k, v shape is [batched_seq_len, hidden_dim]
        # assert batch_size == 1, 'Only batch_size=1 is supported at the moment.'
        attention_mask = torch.ones(query_states.shape[0], query_states.shape[1], dtype=torch.int64, device=query_states.device)
        if self.layer_id == 0:
            print("use_nope: {}".format(self.use_nope))
            print(key_states.shape, attention_mask.shape)
            print(key_states, attention_mask)
            
        compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2 = self._get_compress_k(
            key_states=key_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,   
        )
        
        if self.layer_id == 0:
            print(compressed_k.shape, compressed_cu_seqlens.shape, compressed_k2.shape, compressed_cu_seqlens2.shape)
            print(compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2)
         
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )
        
        if self.layer_id == 0:
            print("line: 1198", query_states.shape, key_states.shape, value_states.shape, indices_q.shape)
            print("line: 1199", query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens)
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens   
        
        
        attn_output_unpad = self.sparse_forward(
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

        attn_output = pad_input(attn_output_unpad, indices_q, 1, query_length)
        
        return attn_output
    
    def sparse_forward(self,
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
        
        if self.layer_id == 0:
            print(topk_idx.shape)
            print(topk_idx)
            
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

    
    
    def _get_compress_k(self, key_states, attention_mask, past_key_value):
        """
        Get compressed key states and corresponding cumulative sequence lengths.
        
        Args:
            key_states: Key states tensor
            cu_seqlens_k: Cumulative sequence lengths for keys
            past_key_value: Past key-value cache
            no_rope_param: Optional parameter containing key states without rope
            
        Returns:
            Tuple of (compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2)
        """
        
        # only support prefill at the moment
        # not to write kv_cache, only support prefill
        unpadded_key_states, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_one_tensor(key_states,attention_mask=attention_mask)
            # Compress the keys
        compressed_k, compressed_cu_seqlens = self.compress_k(unpadded_key_states, cu_seqlens)
        compressed_k2, compressed_cu_seqlens2 = self.compress_k2(unpadded_key_states, cu_seqlens)
        
        # past_key_value.update_compress_k(
        #     compressed_k, self.layer_idx, compressed_cu_seqlens)
        # past_key_value.update_compress_k2(
        #     compressed_k2, self.layer_idx, compressed_cu_seqlens2)
        
        # no_compress_k_list = []
        # # Compute and update no_compress_k
        # for i in range(len(compressed_cu_seqlens)-1):
        #     no_compress_k_start = (compressed_cu_seqlens[i+1]- compressed_cu_seqlens[i]) * self.kernel_stride
            
        #     no_compress_k_list.append(unpadded_key_states[cu_seqlens[i]+no_compress_k_start:cu_seqlens[i+1]].clone())

        # # past_key_value.update_no_compress_k(
        # #     no_compress_k_list, self.layer_idx,kernel_stride=self.kernel_stride, 
        # #     kernel_size=self.kernel_size)
        
        # # Also update no_compress_k2
        # no_compress_k2_list = []
        # for i in range(len(compressed_cu_seqlens2)-1):
        #     no_compress_k2_start = (compressed_cu_seqlens2[i+1]- compressed_cu_seqlens2[i]) * self.kernel_stride * 4
            
        #     no_compress_k2_list.append(unpadded_key_states[cu_seqlens[i]+no_compress_k2_start:cu_seqlens[i+1]].clone())

        # past_key_value.update_no_compress_k2(
        #     no_compress_k2_list, self.layer_idx,kernel_stride=self.kernel_stride*4, 
        #     kernel_size=self.kernel_size*4)
        
        return compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2
    
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
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
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

class MiniCPMDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = MiniCPMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = MiniCPMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, None


class MiniCPMModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiniCPMForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.num_experts = getattr(self.config, "num_experts", 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = MiniCPMForCausalLM
