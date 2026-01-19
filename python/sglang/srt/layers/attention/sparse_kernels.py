import torch
import triton
import triton.language as tl
from functools import lru_cache

@triton.jit
def compress_k_complete_kernel(
    key_ptr,
    cu_seqlens_ptr,
    compressed_k_ptr,
    cu_seqlens_compressed_ptr,
    batch_size,
    max_chunks_per_seq,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complete single-kernel implementation combining chunk calculation and key compression.
    
    This kernel performs all computations entirely within the kernel:
    1. Computes sequence lengths and chunk counts
    2. Computes prefix sums for output positioning
    3. Validates chunk boundaries
    4. Performs mean pooling compression
    5. Updates cumulative sequence lengths
    
    Grid: (batch_size, max_chunks_per_seq, head_num_k)
    Each thread processes one (batch, chunk_in_seq, head) combination.
    
    Args:
        key_ptr: Input key tensor [total_seq_len, head_num_k, head_dim]
        cu_seqlens_ptr: Input cumulative sequence lengths [batch_size + 1]
        compressed_k_ptr: Output compressed key tensor [batch_size * max_chunks_per_seq, head_num_k, head_dim]
        cu_seqlens_compressed_ptr: Output cumulative chunk counts [batch_size + 1]
        batch_size: Number of sequences in batch
        max_chunks_per_seq: Maximum possible chunks per sequence (conservative)
        head_num_k: Number of attention heads
        head_dim: Dimension per head
        kernel_size: Tokens per chunk for compression
        kernel_stride: Stride between chunk starts
        BLOCK_SIZE: Vectorized load/store width
    """
    batch_idx = tl.program_id(0)
    chunk_in_seq = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # ====================================================================
    # PHASE 1: Get sequence info and validate chunk
    # ====================================================================
    
    if batch_idx >= batch_size or head_idx >= head_num_k:
        return
    
    seq_start = tl.load(cu_seqlens_ptr + batch_idx)
    seq_end = tl.load(cu_seqlens_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    
    # Compute how many chunks this sequence actually has
    actual_chunks_in_seq = tl.maximum((seq_len - kernel_size) // kernel_stride + 1, 0)
    
    # Skip if this chunk doesn't exist for this sequence
    if chunk_in_seq >= actual_chunks_in_seq:
        return
    
    chunk_start_pos = seq_start + chunk_in_seq * kernel_stride
    chunk_end_pos = chunk_start_pos + kernel_size
    
    # Additional validation to match original implementation:
    # Only include chunks that are fully contained within the sequence
    if chunk_end_pos > seq_end:
        return
    
    # ====================================================================
    # PHASE 2: Compute prefix sums and global chunk index
    # ====================================================================
    
    # Compute prefix sum of chunks up to this sequence (replicated in each block)
    chunks_before_this_seq = 0
    for s in range(batch_idx):
        s_start = tl.load(cu_seqlens_ptr + s)
        s_end = tl.load(cu_seqlens_ptr + s + 1)
        s_len = s_end - s_start
        s_chunks = tl.maximum((s_len - kernel_size) // kernel_stride + 1, 0)
        chunks_before_this_seq += s_chunks
    
    global_chunk_idx = chunks_before_this_seq + chunk_in_seq
    
    # Update cu_seqlens_compressed (only first thread per sequence)
    if chunk_in_seq == 0 and head_idx == 0:
        # Compute cumulative chunks up to this sequence (inclusive)
        chunks_up_to_this_seq = chunks_before_this_seq + actual_chunks_in_seq
        tl.store(cu_seqlens_compressed_ptr + batch_idx + 1, tl.cast(chunks_up_to_this_seq, tl.int32))
    
    # ====================================================================
    # PHASE 3: Perform mean pooling compression
    # ====================================================================
    
    # Accumulate over all tokens in this chunk
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for token_offset in range(kernel_size):
        token_pos = chunk_start_pos + token_offset
        
        # Compute key memory offset: token_pos * head_num_k * head_dim + head_idx * head_dim
        key_base_offset = token_pos * head_num_k * head_dim + head_idx * head_dim
        
        # Vectorized load of head_dim values
        x = tl.load(
            key_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
            other=0.0
        ).to(tl.float32)
        
        acc += x
    
    # Compute mean over the chunk
    acc = acc / kernel_size
    
    # ====================================================================
    # PHASE 4: Store compressed result
    # ====================================================================
    
    # Store compressed key at global position: global_chunk_idx * head_num_k * head_dim + head_idx * head_dim
    out_offset = global_chunk_idx * head_num_k * head_dim + head_idx * head_dim
    tl.store(
        compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
        acc,
        mask=tl.arange(0, BLOCK_SIZE) < head_dim
    )




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
 ):
    """
    Single-kernel implementation that fuses k computation, key compression,
    key_cache write, and full_compressed_k read for ALL chunks (history + new).

    Grid: (batch_size, max_total_chunks, head_num_k)
    where max_total_chunks = max_chunks_per_seq + max_history_chunks
    - chunk_in_seq in [0, history_chunks_in_seq): process HISTORY chunks
    - chunk_in_seq in [history_chunks_in_seq, total_chunks_in_seq): process NEW chunks

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
    """
    batch_idx = tl.program_id(0)
    chunk_in_seq = tl.program_id(1)
    head_idx = tl.program_id(2)

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

    # Skip if this chunk_in_seq doesn't exist
    if chunk_in_seq >= total_chunks_in_seq:
        return

    # Determine if processing history or new chunks
    is_history_chunk = chunk_in_seq < history_chunks_in_seq

    # Get cumulative positions for this batch
    cu_total_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)

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

        # Validate y is within token_table bounds
        if y >= token_table_cols:
            return

        # Read k_indices from token_table
        k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + y).to(tl.int32)

        # Compute y index in compressed_k_table for new_compressed_k_indices
        # y = new_chunk_idx + history_compress
        compressed_table_y = new_chunk_idx + history_compress

        # Validate y is within compressed_k_table bounds
        if compressed_table_y >= compressed_k_table_cols:
            return

        # Read new_compressed_k_indices from compressed_k_table
        new_compressed_k_indices = tl.load(compressed_k_table_ptr + batch_idx * compressed_k_table_cols + compressed_table_y).to(tl.int32)

        # ====================================================================
        # PHASE 3: Perform mean pooling compression on k
        # ====================================================================

        # Accumulate over all tokens in this chunk
        acc = tl.zeros([head_dim], dtype=tl.float32)

        for token_offset in range(kernel_size):
            # Compute k_indices for this token
            token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride

            # Read k_indices from token_table
            if token_y < token_table_cols:
                token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)
            else:
                token_k_indices = 0

            # Load k from key_cache: key_cache[token_k_indices, head_idx, :]
            key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim

            # Vectorized load of head_dim values
            x = tl.load(
                key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                other=0.0
            ).to(tl.float32)

            acc += x

        # Compute mean over the chunk
        acc = acc / kernel_size

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

                    if token_y < token_table_cols:
                        token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)
                    else:
                        token_k_indices = 0

                    key_base_offset = token_k_indices * head_num_k * head_dim + h * head_dim

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
