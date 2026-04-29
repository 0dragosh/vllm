# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    flashqla_chunk_gated_delta_rule,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("tokens", "num_qk_heads", "num_v_heads"),
    [
        (64, 16, 48),
        (2048, 16, 48),
        (8192, 16, 48),
    ],
)
def test_flashqla_matches_fla_for_qwen36_27b_shapes(
    tokens: int,
    num_qk_heads: int,
    num_v_heads: int,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128

    q = torch.randn(1, tokens, num_qk_heads, head_dim, device=device, dtype=dtype)
    k = F.normalize(
        torch.randn(1, tokens, num_qk_heads, head_dim, device=device, dtype=dtype),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, tokens, num_v_heads, head_dim, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, tokens, num_v_heads, device=device, dtype=dtype))
    beta = torch.sigmoid(
        torch.randn(1, tokens, num_v_heads, device=device, dtype=dtype)
    )
    initial_state = torch.randn(
        1,
        num_v_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    expected_out, expected_state = fla_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    actual_out, actual_state = flashqla_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    torch.testing.assert_close(actual_out, expected_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=3e-2, rtol=3e-2)
