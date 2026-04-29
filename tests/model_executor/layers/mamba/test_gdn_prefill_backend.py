# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule


def make_config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(additional_config={"gdn_prefill_backend": backend})


def make_compilation_config() -> SimpleNamespace:
    return SimpleNamespace(
        custom_ops=["all"],
        disabled_custom_ops=set(),
        enabled_custom_ops=set(),
    )


def test_flashqla_backend_selects_flashqla_forward():
    with (
        patch(
            "vllm.model_executor.custom_op.get_cached_compilation_config",
            return_value=make_compilation_config(),
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn_linear_attn.get_current_vllm_config",
            return_value=make_config("flashqla"),
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn_linear_attn.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn_linear_attn."
            "current_platform.is_device_capability",
            return_value=True,
        ),
    ):
        op = ChunkGatedDeltaRule()

    assert op._forward_method == op.forward_flashqla


def test_triton_backend_selects_native_forward():
    with (
        patch(
            "vllm.model_executor.custom_op.get_cached_compilation_config",
            return_value=make_compilation_config(),
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn_linear_attn.get_current_vllm_config",
            return_value=make_config("triton"),
        ),
    ):
        op = ChunkGatedDeltaRule()

    assert op._forward_method == op.forward_native
