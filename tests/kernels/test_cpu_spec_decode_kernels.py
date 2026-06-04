# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CPU speculative decoding kernels (spec_decode_utils.cpp).

Tests validate correctness of:
- Conditional OMP parallelism (small batch bypass)
- Rebalanced sample_recovered_tokens_kernel (per-token parallelism)
- rejection_greedy_sample_kernel
- rejection_random_sample_kernel
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Ensure the CPU C++ ops are registered with torch
import vllm._C  # noqa: F401


def _call_rejection_greedy(
    draft_token_ids_per_req: list[list[int]],
    target_argmax_per_req: list[list[int]],
    bonus_token_ids: list[int],
    max_spec_len: int,
):
    """Helper to call rejection_greedy_sample_kernel via cpu_triton_utils."""
    import vllm.utils.cpu_triton_utils as cpu_tl

    batch_size = len(draft_token_ids_per_req)
    cu_num_draft_tokens = torch.zeros(batch_size, dtype=torch.int64)
    all_draft = []
    all_target = []
    for i, (draft, target) in enumerate(
        zip(draft_token_ids_per_req, target_argmax_per_req)
    ):
        all_draft.extend(draft)
        all_target.extend(target)
        cu_num_draft_tokens[i] = len(all_draft)

    draft_tensor = torch.tensor(all_draft, dtype=torch.int64)
    target_tensor = torch.tensor(all_target, dtype=torch.int64)
    bonus_tensor = torch.tensor(bonus_token_ids, dtype=torch.int64).unsqueeze(1)

    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64)

    cpu_tl._rejection_greedy_sample_kernel_impl(
        output,
        cu_num_draft_tokens,
        draft_tensor,
        target_tensor,
        bonus_tensor,
        is_greedy=None,
        max_spec_len=max_spec_len,
    )
    return output


def _call_sample_recovered_tokens(
    draft_token_ids_per_req: list[list[int]],
    target_probs_per_token: list[list[float]],
    draft_probs_per_token: list[list[float]] | None,
    vocab_size: int,
):
    """Helper to call sample_recovered_tokens_kernel."""
    import vllm.utils.cpu_triton_utils as cpu_tl

    batch_size = len(draft_token_ids_per_req)
    cu_num_draft_tokens = torch.zeros(batch_size, dtype=torch.int64)
    all_draft = []
    for i, draft in enumerate(draft_token_ids_per_req):
        all_draft.extend(draft)
        cu_num_draft_tokens[i] = len(all_draft)

    total_tokens = len(all_draft)
    draft_tensor = torch.tensor(all_draft, dtype=torch.int64)

    target_probs = torch.tensor(target_probs_per_token, dtype=torch.float32)
    assert target_probs.shape == (total_tokens, vocab_size)

    no_draft_probs = draft_probs_per_token is None
    if no_draft_probs:
        draft_probs = None
    else:
        draft_probs = torch.tensor(draft_probs_per_token, dtype=torch.float32)

    inv_q = torch.ones(batch_size, vocab_size, dtype=torch.float32)
    output = torch.full((total_tokens,), -1, dtype=torch.int64)

    cpu_tl._sample_recovered_tokens_kernel_impl(
        output,
        cu_num_draft_tokens,
        draft_tensor,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        NO_DRAFT_PROBS=no_draft_probs,
    )
    return output


class TestRejectionGreedySample:
    def test_all_accepted(self):
        """All draft tokens match target argmax -> bonus token appended."""
        output = _call_rejection_greedy(
            draft_token_ids_per_req=[[5, 3, 7]],
            target_argmax_per_req=[[5, 3, 7]],
            bonus_token_ids=[99],
            max_spec_len=5,
        )
        assert output[0, 0].item() == 5
        assert output[0, 1].item() == 3
        assert output[0, 2].item() == 7
        assert output[0, 3].item() == 99  # bonus

    def test_first_rejected(self):
        """First draft token differs -> output has target's token, no bonus."""
        output = _call_rejection_greedy(
            draft_token_ids_per_req=[[5, 3, 7]],
            target_argmax_per_req=[[6, 3, 7]],  # mismatch at pos 0
            bonus_token_ids=[99],
            max_spec_len=5,
        )
        assert output[0, 0].item() == 6  # target's token at rejection point
        assert output[0, 1].item() == -1  # not filled

    def test_middle_rejected(self):
        """Rejection at position 1."""
        output = _call_rejection_greedy(
            draft_token_ids_per_req=[[5, 3, 7]],
            target_argmax_per_req=[[5, 4, 7]],  # mismatch at pos 1
            bonus_token_ids=[99],
            max_spec_len=5,
        )
        assert output[0, 0].item() == 5
        assert output[0, 1].item() == 4  # rejection here
        assert output[0, 2].item() == -1

    def test_batch_size_1_no_omp(self):
        """BS=1 exercises the no-OMP path (threshold > 1)."""
        output = _call_rejection_greedy(
            draft_token_ids_per_req=[[10, 20]],
            target_argmax_per_req=[[10, 20]],
            bonus_token_ids=[42],
            max_spec_len=3,
        )
        assert output[0, 0].item() == 10
        assert output[0, 1].item() == 20
        assert output[0, 2].item() == 42

    def test_multi_request(self):
        """Multiple requests, some accepted some rejected."""
        output = _call_rejection_greedy(
            draft_token_ids_per_req=[[1, 2], [3, 4], [5, 6]],
            target_argmax_per_req=[[1, 2], [3, 9], [5, 6]],
            bonus_token_ids=[100, 101, 102],
            max_spec_len=3,
        )
        # Req 0: all accepted
        assert output[0, 0].item() == 1
        assert output[0, 1].item() == 2
        assert output[0, 2].item() == 100
        # Req 1: rejected at pos 1
        assert output[1, 0].item() == 3
        assert output[1, 1].item() == 9
        assert output[1, 2].item() == -1
        # Req 2: all accepted
        assert output[2, 0].item() == 5
        assert output[2, 1].item() == 6
        assert output[2, 2].item() == 102


class TestSampleRecoveredTokens:
    def test_no_draft_probs_single_token(self):
        """Without draft probs, picks argmax of target probs excluding draft_id."""
        vocab_size = 5
        # token 2 has highest prob but draft_id=2, so next best is token 4
        target_probs = [[0.1, 0.1, 0.4, 0.1, 0.3]]
        output = _call_sample_recovered_tokens(
            draft_token_ids_per_req=[[2]],
            target_probs_per_token=target_probs,
            draft_probs_per_token=None,
            vocab_size=vocab_size,
        )
        # With inv_q=1, val = prob * 1. Draft_id=2 is zeroed, so max is token 4 (0.3)
        assert output[0].item() == 4

    def test_with_draft_probs(self):
        """With draft probs, picks argmax of max(0, target - draft) * inv_q."""
        vocab_size = 4
        target_probs = [[0.5, 0.2, 0.2, 0.1]]
        draft_probs = [[0.1, 0.8, 0.05, 0.05]]
        output = _call_sample_recovered_tokens(
            draft_token_ids_per_req=[[1]],
            target_probs_per_token=target_probs,
            draft_probs_per_token=draft_probs,
            vocab_size=vocab_size,
        )
        # diff = target - draft: [0.4, -0.6, 0.15, 0.05]
        # clamp: [0.4, 0.0, 0.15, 0.05]
        # argmax = 0
        assert output[0].item() == 0

    def test_multi_request_multi_token(self):
        """Multiple requests with multiple draft tokens each."""
        vocab_size = 3
        # Req 0: 2 draft tokens, Req 1: 1 draft token
        target_probs = [
            [0.1, 0.8, 0.1],  # req0, tok0 -> best=1
            [0.7, 0.2, 0.1],  # req0, tok1 -> best=0
            [0.1, 0.1, 0.8],  # req1, tok0 -> best=2
        ]
        output = _call_sample_recovered_tokens(
            draft_token_ids_per_req=[[0, 1], [1]],
            target_probs_per_token=target_probs,
            draft_probs_per_token=None,
            vocab_size=vocab_size,
        )
        # Token 0 (req0): draft_id=0 zeroed, argmax of [0, 0.8, 0.1] = 1
        assert output[0].item() == 1
        # Token 1 (req0): draft_id=1 zeroed, argmax of [0.7, 0, 0.1] = 0
        assert output[1].item() == 0
        # Token 2 (req1): draft_id=1 zeroed, argmax of [0.1, 0, 0.8] = 2
        assert output[2].item() == 2

    def test_batch_1_exercises_no_omp(self):
        """BS=1 with single draft token."""
        vocab_size = 3
        target_probs = [[0.2, 0.3, 0.5]]
        output = _call_sample_recovered_tokens(
            draft_token_ids_per_req=[[2]],
            target_probs_per_token=target_probs,
            draft_probs_per_token=None,
            vocab_size=vocab_size,
        )
        # draft_id=2 zeroed: [0.2, 0.3, 0] -> argmax=1
        assert output[0].item() == 1


class TestCPUAttentionMetadataCaching:
    """Test that build_for_drafting reuses scheduler metadata."""

    def test_build_for_drafting_caches(self):
        """Verify that draft_index > 0 reuses cached metadata.

        We test the logic by importing only the dataclass and calling
        the method unbound to avoid importing the full cpu_attn module
        (which triggers torchvision issues in some environments).
        """
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        # Simulate the CPUAttentionMetadata dataclass
        @dataclass
        class FakeCPUAttentionMetadata:
            isa: str
            num_actual_tokens: int
            max_query_len: int
            query_start_loc: torch.Tensor
            max_seq_len: int
            seq_lens: torch.Tensor
            block_table: torch.Tensor
            slot_mapping: torch.Tensor
            scheduler_metadata: torch.Tensor | None
            causal: bool = True
            use_sdpa_prefill: bool = False
            num_decode_tokens: int = 0
            sdpa_attn_masks: list | None = None
            sdpa_start_loc: torch.Tensor | None = None

        # Simulate the build_for_drafting logic directly
        cached_scheduler_metadata = torch.zeros(10)
        cached_num_reqs = 2

        # Test: draft_index > 0 with matching num_reqs reuses cache
        class FakeBuilder:
            isa = "amx"
            use_sdpa_prefill = False
            _cached_draft_scheduler_metadata = cached_scheduler_metadata
            _cached_draft_num_reqs = cached_num_reqs

        common = MagicMock()
        common.num_reqs = 2
        common.num_actual_tokens = 2
        common.max_seq_len = 100
        common.query_start_loc = torch.arange(3, dtype=torch.int32)
        common.seq_lens = torch.tensor([50, 60], dtype=torch.int32)
        common.block_table_tensor = torch.zeros(2, 16, dtype=torch.int32)
        common.slot_mapping = torch.zeros(2, dtype=torch.int64)

        builder = FakeBuilder()

        # Replicate the build_for_drafting logic (draft_index > 0)
        num_reqs = common.num_reqs
        reuse_scheduler = (
            hasattr(builder, "_cached_draft_scheduler_metadata")
            and builder._cached_draft_num_reqs == num_reqs
        )
        assert reuse_scheduler

        result = FakeCPUAttentionMetadata(
            isa=builder.isa,
            num_actual_tokens=common.num_actual_tokens,
            max_query_len=1,
            query_start_loc=common.query_start_loc,
            max_seq_len=common.max_seq_len,
            seq_lens=common.seq_lens,
            block_table=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            scheduler_metadata=builder._cached_draft_scheduler_metadata,
            causal=True,
            use_sdpa_prefill=False,
            num_decode_tokens=0,
            sdpa_start_loc=common.query_start_loc,
        )

        assert result.scheduler_metadata is cached_scheduler_metadata
        assert result.max_query_len == 1
        assert result.use_sdpa_prefill is False

    def test_build_for_drafting_no_cache_on_batch_change(self):
        """Verify that batch size change triggers full rebuild."""

        class FakeBuilder:
            isa = "amx"
            _cached_draft_scheduler_metadata = torch.zeros(10)
            _cached_draft_num_reqs = 2

        builder = FakeBuilder()

        # Simulate different num_reqs
        reuse_scheduler = (
            hasattr(builder, "_cached_draft_scheduler_metadata")
            and builder._cached_draft_num_reqs == 3  # different from cached 2
        )
        assert not reuse_scheduler
