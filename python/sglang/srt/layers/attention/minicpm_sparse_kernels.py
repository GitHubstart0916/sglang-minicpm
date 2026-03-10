import torch
import triton
import triton.language as tl
from functools import lru_cache
import torch.nn.functional as F
import math

# TODO. Now only page size == 1 is supported. Consider extend to page size > 1
@triton.jit
def compress_k_complete_kernel_new(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
    page_size: tl.constexpr,
 ):
    """
    Single-kernel implementation that fuses k computation, key compression,
    key_cache write, and full_compressed_k read for ALL chunks (history + new).

    Grid: (batch_size, min(max_total_chunks, max_grid_chunks), head_num_k)
    where max_total_chunks = max_chunks_per_seq + max_history_chunks
    - chunk_in_seq in [0, history_chunks_in_seq): process HISTORY chunks
    - chunk_in_seq in [history_chunks_in_seq, total_chunks_in_seq): process NEW chunks
    
    If total_chunks > max_grid_chunks, each thread block loops to handle multiple chunks.

    Each thread processes one (batch, chunk_in_seq, head) combination.
    Only head=0 threads write to key_cache or full_compressed_k to avoid redundant writes.

    Args:
        key_cache_ptr: Input key cache tensor [total_tokens, head_num_k, head_dim]
        token_table_ptr: Token table [batch_size, token_table_cols]
        cu_new_k_token_nums_ptr: Cumulative new token nums [batch_size + 1]
        history_compress_k_token_nums_ptr: History compressed token nums [batch_size]
        k_stride: Stride for k computation
        compressed_k_table_ptr: Compressed k table [batch_size, compressed_k_table_cols]
        cu_new_compress_k_token_nums_ptr: Cumulative new compressed token nums [batch_size + 1]
        cu_total_compress_k_token_nums_ptr: Cumulative total compressed token nums [batch_size + 1]
        total_compress_k_token_nums_ptr: Total compressed token nums per batch [batch_size]
        full_compressed_k_ptr: Output buffer [total_compressed_tokens, head_num_k, head_dim]
        batch_size: Number of sequences in batch
        max_chunks_per_seq: Maximum possible NEW chunks per sequence
        token_table_cols: Number of columns in token_table
        compressed_k_table_cols: Number of columns in compressed_k_table
        head_num_k: Number of attention heads
        head_dim: Dimension per head
        kernel_size: Tokens per chunk for compression
        kernel_stride: Stride between chunk starts
        BLOCK_SIZE: Vectorized load/store width
        max_grid_chunks: Maximum grid dimension for chunks (kernel loops if more chunks needed)
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Total number of chunks this thread block needs to process
    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return

    # ====================================================================
    # PHASE 0: Determine chunk type and boundaries
    # ====================================================================

    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)

    # Compute how many NEW chunks this sequence actually has
    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size,
        (new_k_count - kernel_size) // kernel_stride + 1,
        0
    )

    # Total chunks = history + new
    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    # Get cumulative positions for this batch
    cu_total_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)

    # ====================================================================
    # LOOP: Handle multiple chunks per thread block if needed
    # ====================================================================
    
    # Iterate over all chunks assigned to this thread block
    chunk_in_seq = grid_chunk_idx
    
    while chunk_in_seq < total_chunks_in_seq:
        # Determine if processing history or new chunks
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: Process HISTORY chunks
            # ====================================================================

            # chunk_in_seq in [0, history_compress) -> history chunk index
            history_chunk_idx = chunk_in_seq

            # Compute output position in full_compressed_k: cu_total_start + history_chunk_idx
            global_full_idx = cu_total_start + history_chunk_idx

            # Read from compressed_k_table: indices at y = history_chunk_idx
            full_compressed_idx = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + history_chunk_idx).to(tl.int32)

            # Read from key_cache and store to full_compressed_k output
            key_cache_offset = full_compressed_idx * head_num_k * head_dim

            if head_idx == 0:
                for h in range(head_num_k):
                    head_offset = key_cache_offset + h * head_dim

                    x = tl.load(
                        key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                        other=0.0
                    ).to(tl.float32)

                    out_offset = global_full_idx * head_num_k * head_dim + h * head_dim
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        x,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim
                    )

        else:
            # ====================================================================
            # PHASE 2: Process NEW chunks
            # ====================================================================

            # chunk_in_seq in [history_compress, total_chunks_in_seq) -> new chunk index
            new_chunk_idx = chunk_in_seq - history_chunks_in_seq

            # Compute y index in token_table for this new chunk
            # y = new_chunk_idx * kernel_stride + history_compress * k_stride
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            # Use nested if instead of continue (Triton doesn't support continue)
            if y < token_table_cols * page_size:
                # Compute y index in compressed_k_table for new_compressed_k_indices
                # y = new_chunk_idx + history_compress
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    # Read new_compressed_k_indices from compressed_k_table
                    new_compressed_k_indices = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + compressed_table_y).to(tl.int32)

                    # ====================================================================
                    # PHASE 4: Store compressed result to key_cache (head 0 only)
                    # ====================================================================

                    if head_idx == 0:
                        # Compute offset in key_cache for this chunk
                        key_cache_offset = new_compressed_k_indices * head_num_k * head_dim

                        # Store all heads (iterate through all heads and compute/store each)
                        for h in range(head_num_k):
                            head_acc = tl.zeros([head_dim], dtype=tl.float32)

                            for token_offset in range(kernel_size):
                                token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride
                                token_y_page_id = token_y // page_size
                                token_y_in_page = token_y % page_size

                                if token_y < token_table_cols * page_size:
                                    page_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y_page_id).to(tl.int32)
                                else:
                                    page_k_indices = 0

                                # primary key_cache layout: [num_blocks, head_num_k, page_size, head_dim]
                                key_base_offset = page_k_indices * page_size * head_num_k * head_dim + h * page_size * head_dim + token_y_in_page * head_dim

                                x = tl.load(
                                    key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                                    mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                                    other=0.0
                                ).to(tl.float32)

                                head_acc += x

                            head_acc = head_acc / kernel_size

                            # Store this head
                            head_offset = key_cache_offset + h * head_dim
                            tl.store(
                                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                                head_acc,
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim
                            )

                    # ====================================================================
                    # PHASE 5: Read full_compressed_k from key_cache for NEW chunks (head 0 only)
                    # ====================================================================

                    if head_idx == 0:
                        # Compute output position in full_compressed_k: cu_total_start + history_compress + new_chunk_idx
                        global_full_idx = cu_total_start + history_compress + new_chunk_idx

                        # Read full_compressed_k_indices from compressed_k_table
                        full_table_y = history_compress + new_chunk_idx
                        full_compressed_idx = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + full_table_y).to(tl.int32)

                        # Read from key_cache and store to full_compressed_k output buffer
                        key_cache_offset = full_compressed_idx * head_num_k * head_dim

                        # Store all heads
                        for h in range(head_num_k):
                            head_offset = key_cache_offset + h * head_dim

                            x = tl.load(
                                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                                other=0.0
                            ).to(tl.float32)

                            out_offset = global_full_idx * head_num_k * head_dim + h * head_dim
                            tl.store(
                                full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                                x,
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim
                            )
        
        # Move to next chunk for this thread block
        chunk_in_seq += chunk_stride


@triton.jit
def compress_k_complete_kernel_new_padded(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
    page_size: tl.constexpr,
):
    """
    Padded layout version: stores compressed keys in batch-major order.
    
    Output layout: full_compressed_k[batch_idx * max_chunks_per_seq + chunk_idx]
    This allows using reshape() to view per-batch data for debugging.
    
    Grid: (batch_size, min(max_total_chunks, max_grid_chunks), head_num_k)
    where max_total_chunks = max_chunks_per_seq + max_history_chunks
    
    If total_chunks > max_grid_chunks, each thread block loops to handle multiple chunks.
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Total number of chunks this thread block needs to process
    # Each thread block handles: grid_chunk_idx, grid_chunk_idx + max_grid_chunks, grid_chunk_idx + 2*max_grid_chunks, ...
    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return

    # ====================================================================
    # PHASE 0: Determine chunk type and boundaries
    # ====================================================================

    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)

    # Compute how many NEW chunks this sequence actually has
    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size,
        (new_k_count - kernel_size) // kernel_stride + 1,
        0
    )

    # Total chunks = history + new
    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    # ====================================================================
    # LOOP: Handle multiple chunks per thread block if needed
    # ====================================================================
    
    # Iterate over all chunks assigned to this thread block
    # chunk_in_seq = grid_chunk_idx, grid_chunk_idx + chunk_stride, grid_chunk_idx + 2*chunk_stride, ...
    chunk_in_seq = grid_chunk_idx
    
    while chunk_in_seq < total_chunks_in_seq:
        # Skip if this chunk_in_seq doesn't exist
        # (This check is now inside the loop)
        
        # Determine if processing history or new chunks
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: Process HISTORY chunks (PADDED LAYOUT)
            # ====================================================================

            history_chunk_idx = chunk_in_seq

            # PADDED: Store at batch-major position
            global_full_idx = batch_idx * max_chunks_per_seq + history_chunk_idx

            # Read from compressed_k_table
            full_compressed_idx = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + history_chunk_idx).to(tl.int32)

            # Read from key_cache and store to full_compressed_k output
            key_cache_offset = full_compressed_idx * head_num_k * head_dim

            if head_idx == 0:
                for h in range(head_num_k):
                    head_offset = key_cache_offset + h * head_dim

                    x = tl.load(
                        key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                        other=0.0
                    ).to(tl.float32)

                    out_offset = global_full_idx * head_num_k * head_dim + h * head_dim
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        x,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim
                    )

        else:
            # ====================================================================
            # PHASE 2: Process NEW chunks
            # ====================================================================

            new_chunk_idx = chunk_in_seq - history_chunks_in_seq
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            # Use nested if instead of continue (Triton doesn't support continue)
            if y < token_table_cols * page_size:
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    new_compressed_k_indices = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + compressed_table_y).to(tl.int32)

                    # ====================================================================
                    # PHASE 4: Store compressed result to key_cache (head 0 only)
                    # ====================================================================

                    if head_idx == 0:
                        key_cache_offset = new_compressed_k_indices * head_num_k * head_dim

                        for h in range(head_num_k):
                            head_acc = tl.zeros([head_dim], dtype=tl.float32)

                            for token_offset in range(kernel_size):
                                token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride
                                token_y_page_id = token_y // page_size
                                token_y_in_page = token_y % page_size

                                if token_y < token_table_cols * page_size:
                                    page_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y_page_id).to(tl.int32)
                                else:
                                    page_k_indices = 0

                                # key_cache layout: [num_blocks, head_num_k, page_size, head_dim]
                                key_base_offset = page_k_indices * page_size * head_num_k * head_dim + h * page_size * head_dim + token_y_in_page * head_dim

                                x = tl.load(
                                    key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                                    mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                                    other=0.0
                                ).to(tl.float32)

                                head_acc += x

                            head_acc = head_acc / kernel_size

                            head_offset = key_cache_offset + h * head_dim
                            tl.store(
                                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                                head_acc,
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim
                            )

                    # ====================================================================
                    # PHASE 5: Read full_compressed_k from key_cache (PADDED LAYOUT)
                    # ====================================================================

                    if head_idx == 0:
                        # PADDED: Store at batch-major position
                        global_full_idx = batch_idx * max_chunks_per_seq + history_compress + new_chunk_idx

                        full_table_y = history_compress + new_chunk_idx
                        full_compressed_idx = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + full_table_y).to(tl.int32)

                        key_cache_offset = full_compressed_idx * head_num_k * head_dim

                        for h in range(head_num_k):
                            head_offset = key_cache_offset + h * head_dim

                            x = tl.load(
                                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                                other=0.0
                            ).to(tl.float32)

                            out_offset = global_full_idx * head_num_k * head_dim + h * head_dim
                            tl.store(
                                full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                                x,
                                mask=tl.arange(0, BLOCK_SIZE) < head_dim
                            )
        
        # Move to next chunk for this thread block
        chunk_in_seq += chunk_stride


"""Fused CUDA kernel for sparse_page_table to flashinfer format conversion.

This module provides a CUDA graph compatible conversion from MiniCPM's
sparse_page_table format to FlashInfer's kv_indices + kv_indptr format.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl


# Environment variable to select implementation
# Set USE_TRITON_KERNEL=1 to use Triton (CUDA graph compatible)
# Set USE_TRITON_KERNEL=0 to use PyTorch reference (slower, not CUDA graph compatible)
# Default is "1" - always use Triton kernel for CUDA graph compatibility
USE_TRITON_KERNEL = os.environ.get("USE_TRITON_KERNEL", "1") == "1"

# Environment variable to enable comparison between PyTorch and Triton implementations
# Set COMPARE_PYTORCH_TRITON=1 to validate Triton outputs against PyTorch reference
_COMPARISON_ENABLED = os.environ.get("COMPARE_PYTORCH_TRITON", "0") == "1"


#
# Alternative: Two-kernel approach for better performance with large batches
# Kernel 1: Compute cumulative sum (parallel scan)
# Kernel 2: Flatten and fill


@triton.jit
def cumsum_kernel(
    cache_seqlens_ptr,
    kv_indptr_ptr,
    sparse_bs: tl.constexpr,
):
    """Compute cumulative sum using parallel scan algorithm."""
    # Simple sequential implementation for now
    # TODO: Implement parallel scan for better performance
    cumsum = 0
    tl.store(kv_indptr_ptr, 0)

    for i in range(sparse_bs):
        val = tl.load(cache_seqlens_ptr + i)
        cumsum += val
        tl.store(kv_indptr_ptr + i + 1, cumsum)


@triton.jit
def flatten_and_fill_kernel(
    sparse_page_table_ptr,
    cache_seqlens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    kv_last_page_len_ptr,
    max_sparse_tokens: tl.constexpr,
    sparse_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """Flatten sparse_page_table and fill kv_last_page_len."""
    pid = tl.program_id(axis=0)

    if pid >= sparse_bs:
        return

    # Get offset and num_valid
    offset = tl.load(kv_indptr_ptr + pid)
    num_valid = tl.load(cache_seqlens_ptr + pid)

    # Copy valid entries
    num_loops = tl.cdiv(num_valid, BLOCK_SIZE)
    for i in range(num_loops):
        idx = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_valid

        src_idx = pid * max_sparse_tokens + idx
        data = tl.load(sparse_page_table_ptr + src_idx, mask=mask, other=0)

        dst_idx = offset + idx
        tl.store(kv_indices_ptr + dst_idx, data, mask=mask)

    # Fill kv_last_page_len
    tl.store(kv_last_page_len_ptr + pid, 1)


def convert_sparse_to_flashinfer_two_kernel(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
):
    """Two-kernel version for potentially better performance."""
    sparse_bs = cache_seqlens.shape[0]
    max_sparse_tokens = sparse_page_table.shape[1]

    # Kernel 1: Compute cumulative sum
    cumsum_kernel[(1,)](
        cache_seqlens,
        kv_indptr,
        sparse_bs=sparse_bs,
    )

    # Kernel 2: Flatten and fill
    BLOCK_SIZE = 256
    flatten_and_fill_kernel[(sparse_bs,)](
        sparse_page_table,
        cache_seqlens,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_sparse_tokens=max_sparse_tokens,
        sparse_bs=sparse_bs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return kv_indptr, kv_indices, kv_last_page_len


# ============================================================================
# PyTorch Reference Implementation (for testing and fallback)
# ============================================================================


def convert_sparse_to_flashinfer_pytorch(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for sparse_page_table conversion.

    This is the reference implementation used for testing and verification.
    It is NOT CUDA graph compatible due to intermediate allocations.

    Args:
        sparse_page_table: [sparse_bs, max_sparse_tokens] - Valid entries at start
        cache_seqlens: [sparse_bs] - Number of valid entries per row
        kv_indptr: Pre-allocated [sparse_bs + 1] buffer for output
        kv_indices: Pre-allocated [sparse_bs * max_sparse_tokens] buffer for output
        kv_last_page_len: Pre-allocated [sparse_bs] buffer for output

    Returns:
        Tuple of (kv_indptr, kv_indices, kv_last_page_len) - modified in-place
    """
    sparse_bs = cache_seqlens.shape[0]

    # Compute cumulative sum for kv_indptr
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(cache_seqlens, dim=0)

    # Flatten sparse_page_table based on cache_seqlens
    idx = 0
    for i in range(sparse_bs):
        num_valid = cache_seqlens[i].item()
        if num_valid > 0:
            kv_indices[idx : idx + num_valid] = sparse_page_table[i, :num_valid]
            idx += num_valid

    # Fill kv_last_page_len with ones
    kv_last_page_len.fill_(1)

    return kv_indptr, kv_indices, kv_last_page_len


# ============================================================================
# Unified Interface
# ============================================================================


def convert_sparse_page_table_to_flashinfer(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sparse_page_table to FlashInfer format.

    This is the main entry point that selects between PyTorch reference
    implementation and Triton kernel based on USE_TRITON_KERNEL env var.

    Args:
        sparse_page_table: [sparse_bs, max_sparse_tokens] - Valid entries at start
        cache_seqlens: [sparse_bs] - Number of valid entries per row
        kv_indptr: Pre-allocated [sparse_bs + 1] buffer for output
        kv_indices: Pre-allocated [sparse_bs * max_sparse_tokens] buffer for output
        kv_last_page_len: Pre-allocated [sparse_bs] buffer for output

    Returns:
        Tuple of (kv_indptr, kv_indices, kv_last_page_len) - modified in-place

    """
    if True:
        return convert_sparse_to_flashinfer_two_kernel(
            sparse_page_table,
            cache_seqlens,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
        )
    else:
        return convert_sparse_to_flashinfer_pytorch(
            sparse_page_table,
            cache_seqlens,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
        )

def get_sparse_block_table(
    topk_idx, block_table, token_to_bs, 
    token_pos_in_bs, seqlen_k, topk, page_size,
    sparse_block_size):
    
    token_num = topk_idx.shape[1]
    max_num_blocks = block_table.shape[1]
    head_group = topk_idx.shape[0]
    num_pages_per_block = sparse_block_size // page_size
    grid = (token_num,)
    out_block_table = torch.zeros((token_num, head_group, topk * num_pages_per_block),
                                  device=topk_idx.device,
                                  dtype=topk_idx.dtype)

    get_sparse_block_table_kernel[grid](
        topk_idx, block_table,
        token_to_bs, token_pos_in_bs,
        seqlen_k, out_block_table,
        max_num_blocks, token_num,
        page_size, num_pages_per_block,
        topk, head_group, sparse_block_size
    )
    return out_block_table

@triton.jit
def get_sparse_block_table_kernel(
    topk_idx_ptr, block_table_ptr,
    token_to_bs_ptr, token_pos_in_bs_ptr,
    seqlen_k_ptr, out_ptr,
    max_num_blocks, token_num,
    page_size: tl.constexpr,
    num_pages_per_block: tl.constexpr,
    TOPK: tl.constexpr, 
    HEAD_GROUP: tl.constexpr, 
    SPARSE_BLOCK_SIZE: tl.constexpr
):
    token_idx = tl.program_id(0)
    if token_idx >= token_num:
        return

    bs = tl.load(token_to_bs_ptr + token_idx)
    pos_in_bs = tl.load(token_pos_in_bs_ptr + token_idx)
    seqlen_k_bs = tl.load(seqlen_k_ptr + bs)

    page_offsets = tl.arange(0, num_pages_per_block)

    # Unroll head_group loop
    for head_group_idx in range(HEAD_GROUP):
        # Unroll topk loop
        for topk_idx_in_head in range(TOPK):
            # sparse block idx
            sparse_block_ptr = topk_idx_ptr + head_group_idx * token_num * TOPK + token_idx * TOPK + topk_idx_in_head
            sparse_block_idx = tl.load(sparse_block_ptr)

            out_base = (token_idx * HEAD_GROUP * TOPK * num_pages_per_block
                        + head_group_idx * TOPK * num_pages_per_block
                        + topk_idx_in_head * num_pages_per_block)

            # mask for negative sparse block
            mask_valid_block = sparse_block_idx >= 0

            # vectorized page calculation
            token_idx_in_batch = sparse_block_idx * SPARSE_BLOCK_SIZE + page_offsets * page_size
            page_idx_in_batch = token_idx_in_batch // page_size

            # mask for valid page
            mask_page = (token_idx_in_batch < seqlen_k_bs) & (token_idx_in_batch < pos_in_bs)
            mask = mask_valid_block & mask_page

            # vectorized load, store
            page_vals = tl.load(block_table_ptr + bs * max_num_blocks + page_idx_in_batch, mask=mask, other=0)
            page_vals = HEAD_GROUP * page_vals + head_group_idx
            tl.store(out_ptr + out_base + page_offsets, page_vals, mask=mask)
            
def infllmv2_attn_stage1_prefill_ascend(
    q, k, cu_seqlen_q, cu_seqlen_k,
    max_seqlen_k, kernel_stride, causal = True,
):
    """
    q: [tokens_q, num_heads, head_dim]
    k: [tokens_k, nheads_k, head_dim]
    cu_seqlen_q: [batch+1], cumulative sequence length of q
    cu_seqlen_k: [batch+1], cumulative sequence length of k
    max_seqlen_k: int, max sequence length of k
    kernel_stride: int, kernel stride when computing k
    """
    total_tokens, nheads, head_dim = q.shape
    batch_size = len(cu_seqlen_q) - 1
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k
    
    output = torch.zeros(nheads_k, total_tokens, max_seqlen_k, device=q.device, dtype=q.dtype)
    scale = 1.0 / math.sqrt(head_dim)

    for b in range(batch_size):
        start_q = cu_seqlen_q[b]
        end_q = cu_seqlen_q[b+1]
        start_k = cu_seqlen_k[b]
        end_k = cu_seqlen_k[b+1]

        q_b = q[start_q:end_q]         # [seq_len_q_b, nheads, head_dim]
        k_b = k[start_k:end_k]         # [seq_len_k_b, nheads_k, head_dim]
        k_b = k_b.repeat_interleave(nheads_per_group, dim=1).reshape(-1, nheads, head_dim) # [seq_len_k_b, nheads, head_dim]
        
        seq_len_q_b = end_q - start_q
        seq_len_k_b = end_k - start_k

        q_b_t = q_b.transpose(0,1)     # [nheads, seq_len_q, head_dim]
        k_b_t = k_b.transpose(0,1)     # [nheads, seq_len_k, head_dim]
        
        # Q·K^T
        scores = torch.bmm(q_b_t, k_b_t.transpose(1,2)) * scale  # [nheads, seq_len_q, seq_len_k]

        # causal mask
        if causal:
            q_idx = torch.arange(seq_len_q_b, device=scores.device)
            q_compress_idx = ((q_idx - kernel_stride + 1) // kernel_stride) + seq_len_k_b - (seq_len_q_b - kernel_stride + 1) // kernel_stride
            q_compress_idx = q_compress_idx.clamp(0, seq_len_k_b)
            mask = [[0] * q_compress_idx[i] + [1] * (seq_len_k_b - q_compress_idx[i]) for i in range(seq_len_q_b)]
            mask = torch.tensor(mask, dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(mask, float('-inf'))

        # softmax
        probs = F.softmax(scores, dim=-1)  # [nheads, seq_len_q, seq_len_k]
        
        # nheads_per_group reduction
        probs = probs.reshape(nheads_k, nheads_per_group, seq_len_q_b, seq_len_k_b).sum(dim=1)
        
        probs = torch.where(torch.isnan(probs), 0, probs)

        output[:, start_q:end_q, :seq_len_k_b] = probs

    return output

# def infllmv2_attn_stage1_prefill_ascend(
#     q, k, cu_seqlen_q, cu_seqlen_k,
#     max_seqlen_k, kernel_stride, causal = True,
# ):
#     """
#     q: [tokens_q, num_heads, head_dim]
#     k: [tokens_k, nheads_k, head_dim]
#     cu_seqlen_q: [batch+1], cumulative sequence length of q
#     cu_seqlen_k: [batch+1], cumulative sequence length of k
#     max_seqlen_k: int, max sequence length of k
#     kernel_stride: int, kernel stride when computing k
#     """
#     print(q)
#     print(k)
#     total_tokens, nheads, head_dim = q.shape
#     batch_size = len(cu_seqlen_q) - 1
#     nheads_k = k.shape[1]
#     nheads_per_group = nheads // nheads_k
#     q = q.reshape(total_tokens, nheads_k, nheads_per_group, head_dim)
#     q = q.transpose(1, 2).reshape(total_tokens * nheads_per_group, nheads_k, head_dim).contiguous()
    
#     output = torch.zeros((nheads_k, total_tokens, max_seqlen_k), device=q.device, dtype=q.dtype)
#     print(output.shape)

#     for b in range(batch_size):
#         start_q = cu_seqlen_q[b]
#         end_q = cu_seqlen_q[b+1]
#         start_k = cu_seqlen_k[b]
#         end_k = cu_seqlen_k[b+1]

#         q_b = q[start_q:end_q]         # [seq_len_q_b, nheads_k, head_dim]
#         k_b = k[start_k:end_k]         # [seq_len_k_b, nheads_k, head_dim]

#         seq_len_q_b = end_q - start_q
#         seq_len_k_b = end_k - start_k

#         q_b_t = q_b.transpose(0,1)     # [nheads_k, seq_len_q, head_dim]
#         k_b_t = k_b.transpose(0,1)     # [nheads_k, seq_len_k, head_dim]
        
#         # Q·K^T
#         scores = torch.bmm(q_b_t, k_b_t.transpose(1,2))  # [nheads_k, seq_len_q, seq_len_k]

#         # causal mask
#         if causal:
#             q_idx = torch.arange(seq_len_q_b, device=scores.device)
#             q_compress_idx = ((q_idx - kernel_stride + 1) // kernel_stride) + seq_len_k_b - (seq_len_q_b - kernel_stride + 1) // kernel_stride
#             q_compress_idx = q_compress_idx.clamp(0, seq_len_k_b)
#             mask = [[0] * q_compress_idx[i] + [1] * (seq_len_k_b - q_compress_idx[i]) for i in range(seq_len_q_b)]
#             mask = torch.tensor(mask, dtype=torch.bool, device=scores.device)
#             scores = scores.masked_fill(~mask, float('-inf'))

#         # softmax
#         probs = F.softmax(scores, dim=-1)  # [nheads, seq_len_q, seq_len_k]
        
#         # nheads_per_group reduction
#         probs = probs.reshape(nheads_k, seq_len_q_b // nheads_per_group, nheads_per_group, seq_len_k_b).sum(dim=2)
        
#         probs = torch.where(torch.isnan(probs), 0, probs)

#         print("probs shape")
#         print(probs.shape)
#         print(start_q)
#         print(end_q)
#         print(nheads_per_group)
#         print(start_q // nheads_per_group)
#         print(end_q // nheads_per_group)
#         print(seq_len_k_b)
#         print(probs)
#         output[:, start_q//nheads_per_group:end_q//nheads_per_group, :seq_len_k_b] = probs

#     return output

@triton.jit
def max_pooling_1d_varlen_kernel(
    input_ptr,            # [num_heads, total_q, max_k]
    output_ptr,           # [num_heads, total_q, out_len]
    cu_seqlens_q_ptr,     # [batch+1]
    cu_seqlens_k_ptr,     # [batch+1]
    cache_lens_ptr,       # [batch_size]
    batch_size,
    max_seqlen_k,
    out_len,
    num_heads: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    block_size: tl.constexpr,
    local_blocks: tl.constexpr,
    init_blocks: tl.constexpr,
):
    # grid: (total_q, num_heads)
    bidq_global = tl.program_id(0)  # query index across all batches
    bidh = tl.program_id(1)         # head index

    # find batch_idx
    batch_idx = 0
    q_start = 0
    q_end = 0
    k_start = 0
    k_end = 0
    for b in range(batch_size):
        q_start = tl.load(cu_seqlens_q_ptr + b)
        q_end = tl.load(cu_seqlens_q_ptr + b + 1)
        k_start = tl.load(cu_seqlens_k_ptr + b)
        k_end = tl.load(cu_seqlens_k_ptr + b + 1)
        cond = (bidq_global >= q_start) & (bidq_global < q_end)
        batch_idx = tl.where(cond, b, batch_idx)

    # Local query index within the batch
    bidq_local = bidq_global - q_start
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start
    # Skip if this thread is outside the sequence length
    if bidq_local >= seqlen_q:
        return

    # Calculate input and output pointers
    # Input is packed: [num_heads, total_q, max_k]
    # We need to access the k values for this specific query
    total_q_all = tl.load(cu_seqlens_q_ptr + batch_size)
    in_ptr = input_ptr + bidh * total_q_all * max_seqlen_k + bidq_global * max_seqlen_k
    out_ptr = output_ptr + bidh * total_q_all * out_len + bidq_global * out_len

    # Calculate query block index for masking
    cache_len = tl.load(cache_lens_ptr + batch_idx)
    off_bq = (bidq_local + cache_len) // block_size

    for k in range(0, out_len):
        off_bk = k

        # Check causal + local window mask based on exact criteria from transform_score
        should_mask_inf = (off_bk < init_blocks) | ((off_bq >= off_bk) & (off_bq <= off_bk + local_blocks))

        if should_mask_inf:
            tl.store(out_ptr + k, float('inf'))
        else:
            start = k * stride - padding
            end = start + kernel_size
            start = max(start, 0)
            end = min(end, seqlen_k)
            
            max_val = float('-inf')
            if end > start:
                idxs = start + tl.arange(0, kernel_size)
                mask = idxs < end
                vals = tl.load(in_ptr + idxs, mask=mask, other=float('-inf'))
                max_val = tl.max(vals, axis=0)
            tl.store(out_ptr + k, max_val)
            
def max_pooling_1d_varlen(
    input: torch.Tensor, # [num_heads, total_q, max_k]
    cu_seqlens_q: torch.Tensor, # [batch+1]
    cu_seqlens_k: torch.Tensor, # [batch+1]
    cache_lens: torch.Tensor, # [batch_size]
    max_seqlen_k: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int,
    kernel_stride: int,
    kernel_size: int,
) -> torch.Tensor:
    """
    Variable-length version of max_pooling_1d that handles packed sequences.
    
    Args:
        input: Tensor of shape (num_heads, total_q, max_k) where:
               - total_q is sum of all query sequence lengths
               - max_k is the maximum key sequence length (padded)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys (batch_size + 1,)
        cache_lens: Cache lengths for each sequence in the batch (batch_size,)
        max_seqlen_k: Maximum context length
        local_blocks: Number of local blocks for window attention
        init_blocks: Number of initial blocks to mask with inf
        block_size: Block size
        kernel_stride: kernel_stride for pooling
    
    Returns:
        output: Tensor of shape (num_heads, total_q, out_len)
    """
    max_seqlen_k1 = (max_seqlen_k - kernel_size) // kernel_stride + 1 if max_seqlen_k > kernel_size else 0
    out_len = (max_seqlen_k + block_size - 1) // block_size
    print("max_pooling")
    print(max_seqlen_k1)
    print(out_len)
    
    kernel_stride = block_size // kernel_stride
    kernel_size = kernel_stride + 1
    padding = 1
    
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = input.shape[0]
    total_q = input.shape[1]
    
    output = torch.empty(num_heads, total_q, out_len, device=input.device, dtype=input.dtype)
    
    grid = (total_q, num_heads)
    max_pooling_1d_varlen_kernel[grid](
        input, output, cu_seqlens_q, cu_seqlens_k,
        cache_lens, batch_size, max_seqlen_k1,
        out_len, num_heads, kernel_size, kernel_stride,
        padding, block_size, local_blocks, init_blocks
    )
    return output
    