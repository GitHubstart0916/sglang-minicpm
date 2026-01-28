from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MiniCPMReqToTokenPool, MiniCPMHybridReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


class ChunkCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        self.protected_size_ = 0

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        kv_committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

        if isinstance(self.req_to_token_pool, (MiniCPMReqToTokenPool, MiniCPMHybridReqToTokenPool)):
            kernel_size = self.req_to_token_pool.kernel_size
            kernel_stride = self.req_to_token_pool.kernel_stride

            k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1 if kv_committed_len >= kernel_size else 0
            if k1_total > 0:
                k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[req.req_pool_idx, :k1_total]
                self.token_to_kv_pool_allocator.free(k1_indices)

            k2_kernel_size = kernel_size * 4
            k2_kernel_stride = kernel_stride * 4
            k2_total = (kv_committed_len - k2_kernel_size) // k2_kernel_stride + 1 if kv_committed_len >= k2_kernel_size else 0
            if k2_total > 0:
                k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[req.req_pool_idx, :k2_total]
                self.token_to_kv_pool_allocator.free(k2_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        from sglang.srt.mem_cache.memory_pool import MiniCPMHybridReqToTokenPool
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
        # sparse k1, k2 cache indices
        if isinstance(self.req_to_token_pool, (MiniCPMReqToTokenPool, MiniCPMHybridReqToTokenPool)):
            kernel_size = self.req_to_token_pool.kernel_size
            kernel_stride = self.req_to_token_pool.kernel_stride
            num_tokens = len(req.fill_ids)
            num_tokens_k1 = (num_tokens - kernel_size) // kernel_stride + 1 if num_tokens >= kernel_size else 0
            k1_indices = self.req_to_token_pool.req_to_sparse_k1_token[
                req.req_pool_idx, : num_tokens_k1
            ]
            req.prefix_k1_indices = k1_indices.to(dtype=torch.int64, copy=True)
            k2_kernel_size = kernel_size * 4
            k2_kernel_stride = kernel_stride * 4
            num_tokens_k2 = (num_tokens - k2_kernel_size) // k2_kernel_stride + 1 if num_tokens >= k2_kernel_size else 0
            k2_indices = self.req_to_token_pool.req_to_sparse_k2_token[
                req.req_pool_idx, : num_tokens_k2
            ]
            req.prefix_k2_indices = k2_indices.to(dtype=torch.int64, copy=True)

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def protected_size(self):
        # NOTE: no protected size in chunk cache. Chunk cache's eviction is the same with request's lifecycle.
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(params)

        assert (
            params.sliding_window_size is not None
            or params.attention_chunk_size is not None
        ), "Sliding window size or attention chunk size must be set for SWAChunkCache"

        if (
            params.sliding_window_size is not None
            and params.attention_chunk_size is not None
        ):
            logger.warning(
                "Sliding window size and attention chunk size are both set, use sliding window size for chunk cache eviction."
            )

        self.sliding_window_size = params.sliding_window_size
        self.attention_chunk_size = params.attention_chunk_size
        self.window_size = self.sliding_window_size or self.attention_chunk_size

        self.chunked_prefill_size = params.chunked_prefill_size

    def evict_swa(
        self,
        req: Req,
        prelen: int,
    ):
        if self.sliding_window_size is not None:
            # Sliding window attention (e.g. mimo-v2-flash, gpt-oss)
            new_evicted_seqlen_local = max(
                req.evicted_seqlen_local, prelen - self.sliding_window_size
            )
        elif self.attention_chunk_size is not None:
            # Local attention (e.g. llama4)
            new_evicted_seqlen_local = max(
                req.evicted_seqlen_local,
                prelen // self.attention_chunk_size * self.attention_chunk_size,
            )

        if self.page_size > 1:
            new_evicted_seqlen_local = (
                new_evicted_seqlen_local // self.page_size
            ) * self.page_size

        if new_evicted_seqlen_local > req.evicted_seqlen_local:
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
