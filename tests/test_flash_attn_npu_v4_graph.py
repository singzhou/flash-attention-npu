# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

_device_name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
if "Ascend910" not in _device_name and "Ascend950" not in _device_name:
    pytest.skip("FA4 graph tests require Ascend910 or Ascend950", allow_module_level=True)

from flash_attn_npu_4 import flash_attn_varlen_func, get_scheduler_metadata  # noqa: E402
from tests.common.attention_ref import ref_flash_attention_pair  # noqa: E402
from tests.common.compare import assert_fa_close  # noqa: E402
from tests.common.test_utils import make_random_tensor  # noqa: E402


DATA_TYPE = torch.bfloat16
BATCH_SIZE = 1
NUM_HEADS = 4
NUM_KV_HEADS = 2
Q_SEQLEN = 16
KV_SEQLEN = 128
HEAD_SIZE = 128
BLOCK_SIZE = 128
SCALE = 1.0 / (HEAD_SIZE**0.5)
WINDOW_SIZE = (-1, -1)


def _run_flash_attn(
    query,
    key_cache,
    value_cache,
    cache_seqlens,
    page_table,
    cu_seqlens_q,
    scheduler_metadata,
    is_causal,
):
    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=Q_SEQLEN,
        softmax_scale=SCALE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        return_lse=True,
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_varlen_graph(is_causal):
    query = make_random_tensor((Q_SEQLEN, NUM_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    key_cache = make_random_tensor(
        (BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu"
    )
    value_cache = make_random_tensor(
        (BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu"
    )
    cache_seqlens = torch.tensor([KV_SEQLEN], dtype=torch.int32).npu()
    page_table = torch.tensor([[0]], dtype=torch.int32).npu()
    cu_seqlens_q = torch.tensor([0, Q_SEQLEN], dtype=torch.int32).npu()

    metadata_kwargs = dict(
        max_seqlen_q=Q_SEQLEN,
        max_seqlen_k=KV_SEQLEN,
        nheads=NUM_HEADS,
        nheads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        num_splits=1,
        headdim_v=HEAD_SIZE,
        seqused_k=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        causal=is_causal,
        window_size_left=WINDOW_SIZE[0],
        window_size_right=WINDOW_SIZE[1],
    )
    scheduler_metadata = get_scheduler_metadata(**metadata_kwargs)

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q_SEQLEN, KV_SEQLEN),
            diagonal=KV_SEQLEN - Q_SEQLEN + 1,
        ).bool()
    golden_out_ref, _, golden_out_pt, _ = ref_flash_attention_pair(
        query.cpu().unsqueeze(0),
        key_cache.cpu(),
        value_cache.cpu(),
        SCALE,
        causal_mask,
        DATA_TYPE,
        0.0,
    )
    golden_out_ref = golden_out_ref.squeeze(0)
    golden_out_pt = golden_out_pt.squeeze(0)

    _run_flash_attn(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        scheduler_metadata,
        is_causal,
    )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output_npu, *_ = _run_flash_attn(
            query,
            key_cache,
            value_cache,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            scheduler_metadata,
            is_causal,
        )

    graph.replay()
    torch.npu.synchronize()

    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, name="graph out")


@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("kv_capacity", [128, 4096])
@pytest.mark.parametrize("use_side_stream", [False, True])
def test_metadata_captured_with_forward_reads_updated_lengths(
    head_size, kv_capacity, use_side_stream
):
    """Capture metadata and runtime writes on the default or a caller-owned stream."""
    _check_metadata_graph_replay(head_size, kv_capacity, use_side_stream)


def _check_metadata_graph_replay(head_size, kv_capacity, use_side_stream):
    batch_size, q_len, num_heads, num_kv_heads, page_size = 2, 8, 8, 2, 128
    pages_per_seq = kv_capacity // page_size
    query = torch.randn(batch_size * q_len, num_heads, head_size, dtype=DATA_TYPE, device="npu")
    key = torch.randn(
        batch_size * pages_per_seq,
        page_size,
        num_kv_heads,
        head_size,
        dtype=DATA_TYPE,
        device="npu",
    )
    value = torch.randn_like(key)
    lengths = torch.full((batch_size,), kv_capacity, dtype=torch.int32, device="npu")
    cu_query = torch.arange(batch_size + 1, dtype=torch.int32, device="npu") * q_len
    pages = torch.arange(batch_size * pages_per_seq, dtype=torch.int32, device="npu").view(
        batch_size, pages_per_seq
    )
    scale = head_size**-0.5

    def forward():
        metadata = get_scheduler_metadata(
            max_seqlen_q=q_len,
            max_seqlen_k=kv_capacity,
            nheads=num_heads,
            nheads_kv=num_kv_heads,
            headdim=head_size,
            num_splits=0,
            cu_seqlens_q=cu_query,
            seqused_k=lengths,
            causal=False,
        )
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_query,
            seqused_k=lengths,
            page_table=pages,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_capacity,
            softmax_scale=scale,
            causal=False,
            scheduler_metadata=metadata,
            num_splits=0,
        )

    # Allocate inputs above on the original stream, then consume them on the
    # chosen stream. This also exercises cross-stream input storage ownership.
    current = torch.npu.current_stream()
    stream = torch.npu.Stream() if use_side_stream else current
    if use_side_stream:
        stream.wait_stream(current)
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = forward()

        for length in (kv_capacity, 65, kv_capacity // 2):
            lengths.fill_(length)
            graph.replay()
            actual = captured.clone()
            eager = forward()
            torch.npu.synchronize()
            torch.testing.assert_close(actual, eager, atol=0.02, rtol=0.02)

            query_ref = query.float().cpu().reshape(batch_size, q_len, num_heads, head_size)
            key_ref = key.float().cpu().reshape(batch_size, kv_capacity, num_kv_heads, head_size)
            value_ref = (
                value.float().cpu().reshape(batch_size, kv_capacity, num_kv_heads, head_size)
            )
            key_ref = key_ref[:, :length].repeat_interleave(num_heads // num_kv_heads, dim=2)
            value_ref = value_ref[:, :length].repeat_interleave(num_heads // num_kv_heads, dim=2)
            scores = torch.einsum("bqhd,bkhd->bhqk", query_ref, key_ref) * scale
            reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value_ref)
            torch.testing.assert_close(
                actual.float().cpu(),
                reference.reshape(batch_size * q_len, num_heads, head_size),
                atol=0.02,
                rtol=0.02,
            )
    if use_side_stream:
        current.wait_stream(stream)
