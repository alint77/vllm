# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from torch import nn

import vllm.config
import vllm.distributed
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)


def test_default_loader_passes_tiered_layer_map_to_safetensors(monkeypatch):
    from vllm.model_executor.model_loader import default_loader, tiered_moe_physical

    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        enable_eplb=False,
        data_parallel_size=1,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        tiered_moe_config=SimpleNamespace(enabled=True),
    )
    model_config = SimpleNamespace(
        is_moe=True,
        model="/model",
        hf_config=SimpleNamespace(
            model_type="glm_moe_dsa",
            num_hidden_layers=5,
            num_nextn_predict_layers=1,
        ),
        get_num_experts=lambda: 8,
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        tiered_moe_physical,
        "build_tiered_moe_rank_load_plan",
        lambda config, rank: SimpleNamespace(
            rank_plan=SimpleNamespace(owned_expert_ids_by_layer={3: (4, 5), 4: (4, 5)})
        ),
    )
    loader._init_ep_weight_filter(model_config)

    captured = {}

    def fake_iterator(*args, **kwargs):
        captured.update(kwargs)
        yield "dense.weight", nn.Parameter()

    monkeypatch.setattr(default_loader, "safetensors_weights_iterator", fake_iterator)
    monkeypatch.setattr(
        loader,
        "_prepare_weights",
        lambda *args, **kwargs: ("/model", ["model.safetensors"], True),
    )
    source = DefaultModelLoader.Source("/model", revision=None)

    assert [name for name, _ in loader._get_weights_iterator(source)] == [
        "dense.weight"
    ]
    assert captured["local_expert_ids"] is None
    assert captured["local_expert_ids_by_layer"] == {
        3: (4, 5),
        4: (4, 5),
        5: (),
    }


def test_default_loader_uses_native_ep_filter_for_mtp_draft(monkeypatch):
    from vllm.model_executor.model_loader import tiered_moe_physical

    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        enable_eplb=False,
        data_parallel_size=1,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        tiered_moe_config=SimpleNamespace(enabled=True),
    )
    model_config = SimpleNamespace(
        is_moe=True,
        hf_config=SimpleNamespace(
            model_type="deepseek_mtp",
            first_k_dense_replace=1,
            num_hidden_layers=3,
            num_nextn_predict_layers=1,
        ),
        get_num_experts=lambda: 8,
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        tiered_moe_physical,
        "build_tiered_moe_rank_load_plan",
        lambda *_: pytest.fail("MTP draft must not use the target tier plan"),
    )

    loader._init_ep_weight_filter(model_config)

    assert loader.local_expert_ids is None
    assert loader.local_expert_ids_by_layer == {
        1: (),
        2: (),
        3: (4, 5),
    }
    assert loader.tiered_moe_load_plan is None
