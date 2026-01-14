import torch

import torch.nn.functional as F
from einops import rearrange, repeat
from functools import lru_cache

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
            compress_k1 (torch.Tensor): Compressed key tensor.
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
    
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
    
def _unpad_one_tensor(hidden_states, attention_mask):
    # Unpad the hidden states using the indices
    indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]
    
    # Get the remaining dimensions
    remaining_dims = hidden_states.shape[2:]
    
    # Reshape to (batch_size * seq_len, *remaining_dims)
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
    
    # Apply unpadding using indices
    unpadded_states = index_first_axis(reshaped_states, indices)
    
    return unpadded_states, indices, cu_seqlens, max_seqlen_in_batch

def batched_gather(a, cu_seqlen_q, select):
    # 
    select_bs = len(select)
    select = torch.tensor(select, device='cpu')
    starts = cu_seqlen_q[select]
    ends = cu_seqlen_q[select + 1]
    lengths = ends - starts

    max_len = lengths.max()
    local_offsets = torch.arange(max_len, device=a.device)[None, :]
    mask = local_offsets < lengths[:, None]

    local_offsets = local_offsets.expand(select_bs, -1)[mask]

    starts_expanded = starts.repeat_interleave(lengths)
    index = starts_expanded + local_offsets

    return a[index]

def get_compress_k(key_states, attention_mask, 
                    layer,
                    forward_batch,
                    compress_k1: CompressK,
                    compress_k2: CompressK,
                    batch_id = 0):
    # only support batch size 1 for now
    req_id = forward_batch.req_pool_indices[batch_id]

    if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
        past_token_num = forward_batch.extend_prefix_lens_cpu[batch_id]
    else: # decode
        past_token_num = forward_batch.seq_lens_cpu[batch_id] - 1
    past_compress_k1_token_num = (past_token_num - compress_k1.kernel_size) // compress_k1.kernel_stride + 1 if past_token_num >= compress_k1.kernel_size else 0
    past_compress_k2_token_num = (past_token_num - compress_k2.kernel_size) // compress_k2.kernel_stride + 1 if past_token_num >= compress_k2.kernel_size else 0

    k1_compress_idx_st = 0 if past_compress_k1_token_num == 0 else (past_compress_k1_token_num - 1) * compress_k1.kernel_stride + compress_k1.kernel_stride
    k2_compress_idx_st = 0 if past_compress_k2_token_num == 0 else (past_compress_k2_token_num - 1) * compress_k2.kernel_stride + compress_k2.kernel_stride

    token_num = forward_batch.seq_lens_cpu[batch_id].item()

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

    if k1.shape[1] >= compress_k1.kernel_size:
        attention_mask = torch.ones(k1.shape[0], k1.shape[1], dtype=torch.int64, device=k1.device)
        unpadded_key_states, _, cu_seqlens, _ = _unpad_one_tensor(k1,attention_mask=attention_mask)
        compressed_k1, compressed_cu_seqlens = compress_k1(unpadded_key_states, cu_seqlens)
        seq_len = k1.shape[1]
        new_k1_num = (seq_len - compress_k1.kernel_size) // compress_k1.kernel_stride + 1

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.sparse_k1_loc[
                torch.sum(forward_batch.token_num_sparse_k1_cpu[:batch_id]) 
                : torch.sum(forward_batch.token_num_sparse_k1_cpu[:batch_id + 1])], 
            compressed_k1, compressed_k1, None, None
        )

    if k2.shape[1] >= compress_k2.kernel_size:
        attention_mask = torch.ones(k2.shape[0], k2.shape[1], dtype=torch.int64, device=k2.device)
        unpadded_key_states, _, cu_seqlens, _ = _unpad_one_tensor(k2,attention_mask=attention_mask)
        compressed_k2, compressed_cu_seqlens2 = compress_k2(unpadded_key_states, cu_seqlens)
        seq_len = k2.shape[1]
        new_k2_num = (seq_len - compress_k2.kernel_size) // compress_k2.kernel_stride + 1
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.sparse_k2_loc[torch.sum(forward_batch.token_num_sparse_k2_cpu[:batch_id]) 
                : torch.sum(forward_batch.token_num_sparse_k2_cpu[:batch_id + 1])], 
            compressed_k2, compressed_k2, None, None
        )

    compressed_k1, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
    )
    compressed_k1 = compressed_k1[forward_batch.req_to_token_pool.req_to_sparse_k1_token[forward_batch.req_pool_indices[batch_id]][:past_compress_k1_token_num + new_k1_num]]
    compressed_cu_seqlens = torch.tensor([0, compressed_k1.shape[0]], device=compressed_k1.device, dtype=torch.int32)

    compressed_k2, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
    )
    compressed_k2 = compressed_k2[forward_batch.req_to_token_pool.req_to_sparse_k2_token[forward_batch.req_pool_indices[batch_id]][:past_compress_k2_token_num + new_k2_num]]
    compressed_cu_seqlens2 = torch.tensor([0, compressed_k2.shape[0]], device=compressed_k2.device, dtype=torch.int32)

    return compressed_k1, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2