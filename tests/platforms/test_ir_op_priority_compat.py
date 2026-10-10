# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import asdict, fields

import pytest
from vllm.config.kernel import IrOpPriorityConfig

from vllm_omni.platforms.interface import OmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("sparse_priority", [["native"], ["triton", "native"]])
def test_ir_op_priority_uses_installed_vllm_schema(sparse_priority):
    default = ["vllm_c", "native"]
    overrides = {"rms_norm": ["native"], "gelu_and_mul_sparse": sparse_priority}
    priority = OmniPlatform._build_ir_op_priority(default, **overrides)
    supported = {field.name for field in fields(IrOpPriorityConfig)}
    assert asdict(priority) == {name: overrides.get(name, default) for name in supported}
