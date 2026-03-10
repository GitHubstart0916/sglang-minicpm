from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)
from sglang.srt.utils import get_bool_env_var
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

            if self.use_fia:
                self.k_buffer = []
                self.v_buffer = []
                for i in range(self.layer_num):
                    k_buffer_layer = self.kv_buffer[0][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    v_buffer_layer = self.kv_buffer[1][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    self.k_buffer.append(k_buffer_layer)
                    self.v_buffer.append(v_buffer_layer)

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]

            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer,
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer,
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.head_num, self.head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                slot_indices=loc,
            )

class NPUMHATransposedTokenToKVPool(NPUMHATokenToKVPool):
    # kv cache layout: [num_pages, num_heads, page_size, head_dim]
    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
            
        tokens, nhead, head_size = cache_k.shape
        BLOCK_D = triton.next_power_of_2(head_size)
        grid = (tokens, nhead)
        # reshape_and_cache_kernel[grid](
        #     cache_k,
        #     self.k_buffer[layer_id - self.start_layer],
        #     loc,
        #     tokens, 
        #     nhead, 
        #     head_size,
        #     self.page_size,
        #     cache_k.stride(0), 
        #     cache_k.stride(1), 
        #     cache_k.stride(2),
        #     nhead * self.page_size * head_size,
        #     self.page_size * head_size,
        #     head_size,
        #     1,
        #     BLOCK_D=BLOCK_D,
        # )
        # reshape_and_cache_kernel[grid](
        #     cache_v,
        #     self.v_buffer[layer_id - self.start_layer],
        #     loc,
        #     tokens, 
        #     nhead, 
        #     head_size,
        #     self.page_size,
        #     cache_v.stride(0), 
        #     cache_v.stride(1), 
        #     cache_v.stride(2),
        #     nhead * self.page_size * head_size,
        #     self.page_size * head_size,
        #     head_size,
        #     1,
        #     BLOCK_D=BLOCK_D,
        # )
        reshape_and_cache_kernel[grid](
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            tokens, 
            nhead, 
            head_size,
            self.page_size,
            cache_k.stride(0), 
            cache_k.stride(1), 
            cache_k.stride(2),
            cache_v.stride(0), 
            cache_v.stride(1), 
            cache_v.stride(2),
            nhead * self.page_size * head_size,
            self.page_size * head_size,
            head_size,
            1,
            BLOCK_D=BLOCK_D,
        )


class NPUMLATokenToKVPool(MLATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer]

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id - self.start_layer].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

# @triton.jit
# def reshape_and_cache_kernel(
#     k_ptr,
#     k_cache_ptr,
#     loc_ptr,
#     tokens, nhead, headsize,
#     pagesize,
#     stride_k_token, stride_k_head, stride_k_dim,
#     stride_cache_page, stride_cache_head, stride_cache_in_page, stride_cache_dim,
#     BLOCK_D: tl.constexpr,
# ):
#     token_id = tl.program_id(0)
#     head_id  = tl.program_id(1)

#     if token_id >= tokens or head_id >= nhead:
#         return

#     flat_index = tl.load(loc_ptr + token_id)

#     page_id = flat_index // pagesize
#     token_id_in_page  = flat_index % pagesize

#     k_src = (
#         k_ptr
#         + token_id * stride_k_token
#         + head_id  * stride_k_head
#     )

#     k_dst = (
#         k_cache_ptr
#         + page_id * stride_cache_page
#         + head_id * stride_cache_head
#         + token_id_in_page * stride_cache_in_page
#     )

#     offsets = tl.arange(0, BLOCK_D)
#     mask = offsets < headsize

#     k_val = tl.load(k_src + offsets * stride_k_dim, mask=mask)

#     tl.store(k_dst + offsets * stride_cache_dim, k_val, mask=mask)
    
@triton.jit
def reshape_and_cache_kernel(
    k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    loc_ptr,
    tokens, nhead, headsize,
    pagesize,
    stride_k_token, stride_k_head, stride_k_dim,
    stride_v_token, stride_v_head, stride_v_dim,
    stride_cache_page, stride_cache_head, stride_cache_in_page, stride_cache_dim,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    if token_id >= tokens or head_id >= nhead:
        return

    flat_index = tl.load(loc_ptr + token_id)

    page_id = flat_index // pagesize
    token_id_in_page  = flat_index % pagesize

    k_src = (
        k_ptr
        + token_id * stride_k_token
        + head_id  * stride_k_head
    )

    v_src = (
        v_ptr
        + token_id * stride_v_token
        + head_id  * stride_v_head
    )

    k_dst = (
        k_cache_ptr
        + page_id * stride_cache_page
        + head_id * stride_cache_head
        + token_id_in_page * stride_cache_in_page
    )

    v_dst = (
        v_cache_ptr
        + page_id * stride_cache_page
        + head_id * stride_cache_head
        + token_id_in_page * stride_cache_in_page
    )

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < headsize

    k_val = tl.load(k_src + offsets * stride_k_dim, mask=mask)

    tl.store(k_dst + offsets * stride_cache_dim, k_val, mask=mask)
    
    v_val = tl.load(v_src + offsets * stride_v_dim, mask=mask)
    
    tl.store(v_dst + offsets * stride_cache_dim, v_val, mask=mask)