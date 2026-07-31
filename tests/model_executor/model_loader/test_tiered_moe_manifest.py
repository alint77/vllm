# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for header-only tiered MoE checkpoint manifests."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.entrypoints.tiered_moe_plan import _fixed_allocation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
    CompressedTensorsWNA16MarlinMoEMethod,
)
from vllm.model_executor.model_loader import (
    tiered_moe_execution as tiered_execution_module,
)
from vllm.model_executor.model_loader import tiered_moe_manifest as manifest_module
from vllm.model_executor.model_loader import tiered_moe_streaming as streaming_module
from vllm.model_executor.model_loader.tiered_moe_conversion import (
    OneExpertCheckpointStager,
    is_glm_expert_checkpoint_tensor,
)
from vllm.model_executor.model_loader.tiered_moe_execution import (
    _active_tier_expert_ids,
)
from vllm.model_executor.model_loader.tiered_moe_kv import (
    get_tiered_kv_available_memory,
    get_tiered_kv_memory_tier,
    plan_glm_kv_cache,
)
from vllm.model_executor.model_loader.tiered_moe_machine import (
    load_grace_machine_profile,
)
from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TensorManifestEntry,
    TieredMoECheckpointManifest,
    _make_entry,
    _validate_expert_entries,
    _validate_glm_w4a16_config,
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    build_glm_non_routed_runtime_inventory,
)
from vllm.model_executor.model_loader.tiered_moe_physical import (
    attach_tiered_moe_layer_placement,
    get_tiered_moe_rank_load_plan,
    resolve_layer_expert_placement,
    use_tiered_moe_rank_load_plan,
    validate_tiered_moe_observed_hbm_reserve,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    load_tiered_moe_placement_profile,
)
from vllm.model_executor.model_loader.tiered_moe_planner import (
    LayerExpertPlacement,
    build_layer_expert_ownership_map,
    plan_rank_expert_tiers,
)
from vllm.model_executor.model_loader.tiered_moe_runtime import (
    plan_tiered_glm_runtime_buffers,
)
from vllm.model_executor.model_loader.tiered_moe_scheduler import (
    _replica_route_hash,
    assign_replicated_experts,
    greedy_replica_assignment,
    validate_replicated_routes,
)
from vllm.model_executor.model_loader.tiered_moe_storage import (
    GLM_MARLIN_COMPONENTS,
    GLM_MARLIN_EXPERT_BYTES,
    ExpertTierStorage,
    LayerTieredExpertStorage,
    build_expert_component_views,
    glm_marlin_components,
)
from vllm.model_executor.model_loader.tiered_moe_streaming import (
    TieredMoEExpertLoader,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)


def make_glm_config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "max_position_embeddings": 1_048_576,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "index_topk_pattern": None,
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "actorder": "static",
                        "group_size": 128,
                        "num_bits": 4,
                        "strategy": "group",
                        "symmetric": True,
                        "type": "int",
                    }
                }
            },
        },
    }


def make_auto_round_glm_config() -> dict:
    config = make_glm_config()
    config["quantization_config"] = {
        "bits": 4,
        "data_type": "int",
        "group_size": 64,
        "packing_format": "auto_round:auto_gptq",
        "quant_method": "auto-round",
        "sym": True,
    }
    return config


def test_host_uva_kv_plan_preserves_native_block_count_and_tiers():
    main_spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
    )
    indexer_spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    specs = {
        **{f"main.{index}": main_spec for index in range(78)},
        **{f"indexer.{index}": indexer_spec for index in range(21)},
    }
    group = KVCacheGroupSpec(
        layer_names=list(specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs=specs,
        ),
    )
    vllm_config = SimpleNamespace(
        tiered_moe_config=SimpleNamespace(enabled=True, mla_cache_tier="host_uva"),
        model_config=SimpleNamespace(max_model_len=400_000),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        kv_transfer_config=None,
    )

    available_memory = get_tiered_kv_available_memory(vllm_config, [group])
    assert available_memory == 21_579_452_160
    config = get_kv_cache_config_from_groups(vllm_config, [group], available_memory)

    assert config.num_blocks == 6251
    assert get_tiered_kv_memory_tier(vllm_config, main_spec) == (
        "host_uva",
        "grace_numa_node",
    )
    assert get_tiered_kv_memory_tier(vllm_config, indexer_spec) == (
        "hbm",
        "gpu_rank",
    )
    assert [tensor.memory_tier for tensor in config.kv_cache_tensors].count(
        "host_uva"
    ) == 78
    assert [tensor.memory_tier for tensor in config.kv_cache_tensors].count("hbm") == 21


def test_hbm_kv_plan_preserves_semantic_main_and_indexer_counts():
    main_spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
    )
    indexer_spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    specs = {
        **{f"main.{index}": main_spec for index in range(78)},
        **{f"indexer.{index}": indexer_spec for index in range(21)},
    }
    group = KVCacheGroupSpec(
        layer_names=list(specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs=specs,
        ),
    )
    vllm_config = SimpleNamespace(
        tiered_moe_config=SimpleNamespace(enabled=True, mla_cache_tier="hbm"),
        model_config=SimpleNamespace(max_model_len=400_000),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        kv_transfer_config=None,
    )

    available_memory = get_tiered_kv_available_memory(vllm_config, [group])
    assert available_memory == 21_579_452_160
    config = get_kv_cache_config_from_groups(vllm_config, [group], available_memory)

    assert config.num_blocks == 6251
    assert len(config.kv_cache_tensors) == 99
    assert all(tensor.memory_tier == "hbm" for tensor in config.kv_cache_tensors)
    assert get_tiered_kv_memory_tier(vllm_config, main_spec) == (
        "hbm",
        "gpu_rank",
    )
    assert get_tiered_kv_memory_tier(vllm_config, indexer_spec) == (
        "hbm",
        "gpu_rank",
    )


def test_glm_descriptor_rejects_wrong_topk():
    config = make_glm_config()
    config["num_experts_per_tok"] = 4

    with pytest.raises(ValueError, match="top-k"):
        _validate_glm_w4a16_config(config)


def test_glm_descriptor_accepts_auto_round_w4g64():
    layers, checkpoint_format, group_size = _validate_glm_w4a16_config(
        make_auto_round_glm_config()
    )

    assert layers == tuple(range(3, 78))
    assert checkpoint_format == "auto_round_gptq"
    assert group_size == 64


def test_expert_inventory_includes_every_stored_component():
    entries = []
    shapes = {
        "gate_proj": {
            "weight_packed": (2048, 768),
            "weight_scale": (2048, 48),
            "weight_shape": (2,),
        },
        "up_proj": {
            "weight_packed": (2048, 768),
            "weight_scale": (2048, 48),
            "weight_shape": (2,),
        },
        "down_proj": {
            "weight_packed": (6144, 256),
            "weight_scale": (6144, 16),
            "weight_shape": (2,),
        },
    }
    for projection, components in shapes.items():
        for component, shape in components.items():
            dtype = (
                "I64"
                if component == "weight_shape"
                else ("BF16" if component == "weight_scale" else "I32")
            )
            name = f"model.layers.3.mlp.experts.0.{projection}.{component}"
            entries.append(_make_entry(name, "model.safetensors", dtype, shape))

    checkpoint_bytes, runtime_bytes = _validate_expert_entries(
        tuple(entries), (3,), num_experts=1, checkpoint_format="compressed_tensors"
    )

    assert checkpoint_bytes == 19_464_240
    assert runtime_bytes == 19_464_200


def test_auto_round_expert_inventory_drops_symmetric_zero_points_at_runtime():
    entries = []
    shapes = {
        "gate_proj": {
            "qweight": ((768, 2048), "I32"),
            "qzeros": ((96, 256), "I32"),
            "scales": ((96, 2048), "F16"),
        },
        "up_proj": {
            "qweight": ((768, 2048), "I32"),
            "qzeros": ((96, 256), "I32"),
            "scales": ((96, 2048), "F16"),
        },
        "down_proj": {
            "qweight": ((256, 6144), "I32"),
            "qzeros": ((32, 768), "I32"),
            "scales": ((32, 6144), "F16"),
        },
    }
    for projection, components in shapes.items():
        for component, (shape, dtype) in components.items():
            name = f"model.layers.3.mlp.experts.0.{projection}.{component}"
            entries.append(_make_entry(name, "model.safetensors", dtype, shape))

    checkpoint_bytes, runtime_bytes = _validate_expert_entries(
        tuple(entries), (3,), num_experts=1, checkpoint_format="auto_round_gptq"
    )

    assert checkpoint_bytes == 20_348_928
    assert runtime_bytes == 20_054_024


def test_mtp_experts_are_not_intercepted_by_tiered_streaming():
    name = "model.layers.78.mlp.experts.0.gate_proj.qweight"

    entry = _make_entry(name, "model.safetensors", "I32", (768, 2048))
    assert not entry.is_routed_expert
    assert not is_glm_expert_checkpoint_tensor(name)


def write_tiny_checkpoint(tmp_path, declared_bytes: int = 16):
    tensor_path = tmp_path / "model.safetensors"
    save_file({"dense.weight": torch.arange(4, dtype=torch.int32)}, tensor_path)
    (tmp_path / "config.json").write_text("{}")
    index = {
        "metadata": {"total_size": declared_bytes},
        "weight_map": {"dense.weight": tensor_path.name},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_manifest_uses_header_shapes_for_exact_bytes(tmp_path, monkeypatch):
    write_tiny_checkpoint(tmp_path)
    monkeypatch.setattr(
        manifest_module,
        "_validate_glm_w4a16_config",
        lambda config: ((3,), "compressed_tensors", 128),
    )
    monkeypatch.setattr(
        manifest_module,
        "_validate_expert_entries",
        lambda entries, routed_layers, num_experts, checkpoint_format: (0, 0),
    )

    manifest = build_glm_w4a16_manifest(tmp_path)

    assert manifest.checkpoint_bytes == 16
    assert manifest.non_routed_bytes == 16
    assert manifest.entries[0].shape == (4,)
    assert manifest.entries[0].dtype == "I32"


def test_manifest_rejects_index_byte_mismatch(tmp_path, monkeypatch):
    write_tiny_checkpoint(tmp_path, declared_bytes=15)
    monkeypatch.setattr(
        manifest_module,
        "_validate_glm_w4a16_config",
        lambda config: ((3,), "compressed_tensors", 128),
    )

    with pytest.raises(ValueError, match="total_size mismatch"):
        build_glm_w4a16_manifest(tmp_path)


def make_planner_manifest() -> TieredMoECheckpointManifest:
    return TieredMoECheckpointManifest(
        model_path=Path("/model"),
        config_sha256="config",
        index_sha256="index",
        entries=(),
        checkpoint_bytes=800,
        routed_expert_bytes=800,
        non_routed_bytes=0,
        routed_layers=(3, 4),
        num_experts=4,
        checkpoint_expert_bytes=100,
        runtime_expert_bytes=100,
    )


def test_planner_assigns_exact_bytes_with_even_rotation():
    plan = plan_rank_expert_tiers(
        make_planner_manifest(),
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=650,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"weights": 200, "cache": 100},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
    )

    assert plan.owned_expert_ids == (0, 1)
    assert plan.hot_expert_slots == 2
    assert plan.cold_expert_slots == 2
    assert plan.hot_expert_bytes == 200
    assert plan.cold_expert_bytes == 200
    assert plan.planned_hbm_bytes == 600
    assert plan.layer_placements[0].hot_expert_ids == (0,)
    assert plan.layer_placements[1].hot_expert_ids == (1,)
    assert set(plan.owned_expert_ids_by_layer[3]) == {0, 1}
    assert set(plan.owned_expert_ids_by_layer[4]) == {0, 1}


def test_layer_ownership_map_covers_every_routed_layer():
    ownership = build_layer_expert_ownership_map(
        make_planner_manifest(), ep_size=2, ep_rank=1
    )

    assert ownership == {3: (2, 3), 4: (2, 3)}


def write_placement_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "config_sha256": "config",
                "index_sha256": "index",
                "ep_size": 2,
                "num_experts": 4,
                "routed_layers": [3, 4],
                "owners": [[0, 1, 0, 1], [1, 0, 1, 0]],
                "hot_experts": [[1, 2], [1, 2]],
                "optimizer": "test",
                "training_request_hashes": ["train"],
                "heldout_request_hashes": ["heldout"],
            }
        )
    )


def write_replica_placement_profile(path: Path) -> None:
    write_placement_profile(path)
    data = json.loads(path.read_text())
    data["profile_version"] = 2
    data["secondary_ranks"] = [
        [1, 0, -1, -1],
        [0, 1, -1, -1],
    ]
    path.write_text(json.dumps(data))


def test_trace_placement_profile_drives_owner_and_residency(tmp_path):
    path = tmp_path / "placement.json"
    write_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)

    assert profile.ownership_for_rank(0) == {3: (0, 2), 4: (1, 3)}
    assert profile.hot_for_rank(0) == {3: (2,), 4: (1,)}

    plan = plan_rank_expert_tiers(
        manifest,
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=650,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"fixed": 300},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
        owned_expert_ids_by_layer=profile.ownership_for_rank(0),
        hot_expert_ids_by_layer=profile.hot_for_rank(0),
    )

    assert plan.layer_placements == (
        LayerExpertPlacement(3, (2,), (0,)),
        LayerExpertPlacement(4, (1,), (3,)),
    )


def test_replica_profile_adds_cold_loader_destinations(tmp_path):
    """A v2 profile keeps primary ownership unchanged while adding one
    Grace-only secondary per layer and accounting its exact bytes."""
    path = tmp_path / "placement.json"
    write_replica_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)

    assert profile.replicas_for_rank(0) == {3: (1,), 4: (0,)}
    plan = plan_rank_expert_tiers(
        manifest,
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=650,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"fixed": 300},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
        owned_expert_ids_by_layer=profile.ownership_for_rank(0),
        hot_expert_ids_by_layer=profile.hot_for_rank(0),
        replica_expert_ids_by_layer=profile.replicas_for_rank(0),
    )

    assert plan.layer_placements == (
        LayerExpertPlacement(3, (2,), (0,), (1,)),
        LayerExpertPlacement(4, (1,), (3,), (0,)),
    )
    assert plan.replica_expert_slots == 2
    assert plan.cold_expert_slots == 4
    assert plan.cold_expert_bytes == 400
    assert plan.owned_expert_ids_by_layer == {
        3: (2, 0, 1),
        4: (1, 3, 0),
    }


def test_replica_assignment_off_preserves_primary_tiers():
    placement = LayerExpertPlacement(3, (0,), (2,), (1, 3))

    hot, cold, selected = _active_tier_expert_ids(
        placement,
        ep_rank=0,
        ep_size=2,
        primary_ranks=None,
        secondary_ranks=None,
        assignment="off",
    )

    assert hot == (0,)
    assert cold == (2,)
    assert selected is None


def test_static_secondary_assignment_executes_every_route_once():
    primary_ranks = (0, 1, 0, 1)
    secondary_ranks = (1, 0, -1, 0)
    placements = (
        LayerExpertPlacement(3, (0,), (2,), (1, 3)),
        LayerExpertPlacement(3, (1,), (3,), (0,)),
    )
    active_by_rank = []
    selected_by_rank = []
    for rank, placement in enumerate(placements):
        hot, cold, selected = _active_tier_expert_ids(
            placement,
            ep_rank=rank,
            ep_size=2,
            primary_ranks=primary_ranks,
            secondary_ranks=secondary_ranks,
            assignment="secondary",
        )
        active_by_rank.append(set(hot + cold))
        selected_by_rank.append(selected)

    assert selected_by_rank[0] == selected_by_rank[1] == (1, 0, 0, 0)
    routes = (0, 1, 1, 2, 3, -1)
    for expert_id in routes:
        if expert_id < 0:
            continue
        ranks = [
            rank for rank, active in enumerate(active_by_rank) if expert_id in active
        ]
        assert ranks == [selected_by_rank[0][expert_id]]

    route_outputs = torch.arange(len(routes) * 3, dtype=torch.float32).view(-1, 3)
    baseline = route_outputs[:-1].sum(0)
    replicated = sum(
        (
            route_outputs[[expert_id in active for expert_id in routes]].sum(0)
            for active in active_by_rank
        ),
        torch.zeros(3),
    )
    torch.testing.assert_close(replicated, baseline)


def test_static_secondary_assignment_rejects_missing_local_copy():
    with pytest.raises(ValueError, match="secondary assignment"):
        _active_tier_expert_ids(
            LayerExpertPlacement(3, (0,), (2,), (1,)),
            ep_rank=0,
            ep_size=2,
            primary_ranks=(0, 1, 0, 1),
            secondary_ranks=(1, 0, -1, 0),
            assignment="secondary",
        )


def test_greedy_replica_assignment_balances_predicted_rank_span():
    selected = greedy_replica_assignment(
        route_counts=(2, 1, 0, 1),
        primary_ranks=(0, 1, 0, 1),
        secondary_ranks=(1, 0, -1, 0),
        primary_hot=(True, False, True, False),
        ep_size=2,
    )

    assert selected == (0, 0, -1, 1)


def test_replica_route_check_rejects_divergent_rank_routes(monkeypatch):
    local_routes = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    other_routes = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    reduced_hash = _replica_route_hash(local_routes) + _replica_route_hash(other_routes)
    group = SimpleNamespace(all_reduce=lambda _: reduced_hash)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_ep_group",
        lambda: group,
    )

    with pytest.raises(RuntimeError, match="divergent cross-rank routes"):
        validate_replicated_routes(local_routes, ep_size=2)


def test_replica_route_check_accepts_identical_rank_routes(monkeypatch):
    routes = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    group = SimpleNamespace(all_reduce=lambda value: value * 2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_ep_group",
        lambda: group,
    )

    validate_replicated_routes(routes, ep_size=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_greedy_replica_assignment_executes_every_route_once():
    routes = torch.tensor(
        [[0, 0, 1, 3, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    primary_ranks = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    secondary_ranks = torch.tensor([1, 0, -1, 0], dtype=torch.int32, device="cuda")
    primary_hot = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="cuda")
    expected = greedy_replica_assignment(
        route_counts=(2, 1, 0, 1),
        primary_ranks=primary_ranks.tolist(),
        secondary_ranks=secondary_ranks.tolist(),
        primary_hot=(True, False, True, False),
        ep_size=2,
    )
    active_by_rank = []

    for rank in range(2):
        hot_primary_map = torch.tensor(
            [0, -1, 1, -1] if rank == 0 else [-1, -1, -1, -1],
            dtype=torch.int32,
            device="cuda",
        )
        cold_primary_map = torch.tensor(
            [-1, -1, -1, -1] if rank == 0 else [-1, 0, -1, 1],
            dtype=torch.int32,
            device="cuda",
        )
        cold_physical_map = torch.tensor(
            [-1, 0, -1, 1] if rank == 0 else [0, 1, -1, 2],
            dtype=torch.int32,
            device="cuda",
        )
        hot_output_map = torch.empty(4, dtype=torch.int32, device="cuda")
        cold_output_map = torch.empty(4, dtype=torch.int32, device="cuda")
        selected_ranks = torch.empty(4, dtype=torch.int32, device="cuda")
        assign_replicated_experts(
            routes,
            primary_ranks,
            secondary_ranks,
            primary_hot,
            hot_primary_map,
            cold_primary_map,
            cold_physical_map,
            hot_output_map,
            cold_output_map,
            selected_ranks,
            ep_size=2,
            ep_rank=rank,
            schedule=True,
        )
        torch.accelerator.synchronize()
        assert tuple(selected_ranks.tolist()) == expected
        active_by_rank.append(
            {
                expert
                for expert in range(4)
                if hot_output_map[expert] >= 0 or cold_output_map[expert] >= 0
            }
        )
        assign_replicated_experts(
            routes,
            primary_ranks,
            secondary_ranks,
            primary_hot,
            hot_primary_map,
            cold_primary_map,
            cold_physical_map,
            hot_output_map,
            cold_output_map,
            selected_ranks,
            ep_size=2,
            ep_rank=rank,
            schedule=False,
        )
        torch.testing.assert_close(hot_output_map, hot_primary_map)
        torch.testing.assert_close(cold_output_map, cold_primary_map)
        torch.testing.assert_close(selected_ranks, primary_ranks)

    for expert in (0, 1, 3):
        executing_ranks = [
            rank for rank, active in enumerate(active_by_rank) if expert in active
        ]
        assert executing_ranks == [expected[expert]]


def test_attach_layer_placement_keeps_cross_rank_maps():
    placement = LayerExpertPlacement(3, (0,), (2,), (1, 3))
    profile = SimpleNamespace(
        routed_layers=(3,),
        owners=((0, 1, 0, 1),),
        hot_experts=((0, 2),),
        secondary_ranks=((1, 0, -1, 0),),
        num_experts=4,
    )
    plan = SimpleNamespace(
        rank_plan=SimpleNamespace(ep_rank=0),
        placement_profile=profile,
    )
    layer = SimpleNamespace()

    attach_tiered_moe_layer_placement(
        layer,
        plan,  # type: ignore[arg-type]
        placement,
    )

    assert layer.tiered_moe_placement is placement
    assert layer.tiered_moe_ep_rank == 0
    assert layer.tiered_moe_primary_ranks == (0, 1, 0, 1)
    assert layer.tiered_moe_secondary_ranks == (1, 0, -1, 0)
    assert layer.tiered_moe_primary_hot == (True, False, True, False)


def test_setup_static_secondary_maps_activate_selected_copies(monkeypatch):
    config = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        tiered_moe_config=SimpleNamespace(replica_assignment="secondary"),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        tiered_execution_module,
        "make_wna16_moe_quant_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tiered_execution_module,
        "make_wna16_moe_kernel",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tiered_execution_module,
        "_apply_tier_launch_policy",
        lambda *args: None,
    )

    def tier(expert_ids):
        return SimpleNamespace(
            expert_ids=expert_ids,
            buffer=torch.empty(1, dtype=torch.uint8),
            components={
                "w13_weight_scale": torch.empty(0),
                "w2_weight_scale": torch.empty(0),
            },
        )

    method = SimpleNamespace(experts_cls=object(), moe=object(), is_k_full=True)
    layer = torch.nn.Module()
    layer.global_num_experts = 4
    layer.tiered_moe_ep_rank = 0
    layer.tiered_moe_placement = LayerExpertPlacement(3, (0,), (2,), (1, 3))
    layer.tiered_moe_primary_ranks = (0, 1, 0, 1)
    layer.tiered_moe_secondary_ranks = (1, 0, -1, 0)
    layer.tiered_moe_storage = SimpleNamespace(
        hot=tier((0,)),
        cold=tier((2, 1, 3)),
    )

    tiered_execution_module.setup_tiered_moe_kernels(method, layer, 64, 4)

    assert layer.tiered_hot_expert_map.tolist() == [-1, -1, -1, -1]
    assert layer.tiered_cold_expert_map.tolist() == [-1, 1, 0, 2]
    assert layer.tiered_replica_selected_ranks.tolist() == [1, 0, 0, 0]


def test_setup_greedy_assignment_keeps_writable_tier_maps(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        tiered_moe_config=SimpleNamespace(replica_assignment="greedy"),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        tiered_execution_module,
        "make_wna16_moe_quant_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tiered_execution_module,
        "make_wna16_moe_kernel",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tiered_execution_module,
        "_apply_tier_launch_policy",
        lambda *args: None,
    )

    def tier(expert_ids):
        return SimpleNamespace(
            expert_ids=expert_ids,
            buffer=torch.empty(1, dtype=torch.uint8),
            components={
                "w13_weight_scale": torch.empty(0),
                "w2_weight_scale": torch.empty(0),
            },
        )

    method = SimpleNamespace(experts_cls=object(), moe=object(), is_k_full=True)
    layer = torch.nn.Module()
    layer.global_num_experts = 4
    layer.tiered_moe_ep_rank = 0
    layer.tiered_moe_placement = LayerExpertPlacement(3, (0,), (2,), (1, 3))
    layer.tiered_moe_primary_ranks = (0, 1, 0, 1)
    layer.tiered_moe_secondary_ranks = (1, 0, -1, 0)
    layer.tiered_moe_primary_hot = (True, False, False, False)
    layer.tiered_moe_storage = SimpleNamespace(
        hot=tier((0,)),
        cold=tier((2, 1, 3)),
    )

    tiered_execution_module.setup_tiered_moe_kernels(method, layer, 64, 4)

    assert method.tiered_replica_assignment == "greedy"
    assert method.tiered_overlap_max_tokens == 16
    assert layer.tiered_hot_primary_expert_map.tolist() == [0, -1, -1, -1]
    assert layer.tiered_cold_primary_expert_map.tolist() == [-1, -1, 0, -1]
    assert layer.tiered_cold_physical_expert_map.tolist() == [-1, 1, 0, 2]
    assert layer.tiered_hot_expert_map.tolist() == [-1, -1, -1, -1]
    assert layer.tiered_cold_expert_map.tolist() == [-1, -1, -1, -1]
    assert layer.tiered_replica_selected_ranks.tolist() == [-1, -1, -1, -1]


def test_replica_profile_rejects_secondary_on_primary_rank(tmp_path):
    path = tmp_path / "placement.json"
    write_replica_placement_profile(path)
    data = json.loads(path.read_text())
    data["secondary_ranks"][0][0] = data["owners"][0][0]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="secondary rank matches"):
        load_tiered_moe_placement_profile(path, make_planner_manifest(), ep_size=2)


def test_planner_includes_replicas_in_grace_capacity_check(tmp_path):
    path = tmp_path / "placement.json"
    write_replica_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)

    with pytest.raises(ValueError, match="paired Grace capacity"):
        plan_rank_expert_tiers(
            manifest,
            ep_size=2,
            ep_rank=0,
            hbm_capacity_bytes=650,
            hbm_reserve_bytes=100,
            fixed_hbm_allocations={"fixed": 300},
            host_capacity_bytes=499,
            host_reserve_bytes=100,
            minimum_hbm_reserve_bytes=0,
            minimum_host_reserve_bytes=0,
            owned_expert_ids_by_layer=profile.ownership_for_rank(0),
            hot_expert_ids_by_layer=profile.hot_for_rank(0),
            replica_expert_ids_by_layer=profile.replicas_for_rank(0),
        )


def test_residency_profile_promotes_cold_when_hbm_budget_grows(tmp_path):
    """A larger HBM budget (e.g. DCP shrinking the KV cache) must fill the
    extra slots deterministically instead of failing or idling HBM."""
    path = tmp_path / "placement.json"
    write_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)

    plan = plan_rank_expert_tiers(
        manifest,
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=750,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"fixed": 300},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
        owned_expert_ids_by_layer=profile.ownership_for_rank(0),
        hot_expert_ids_by_layer=profile.hot_for_rank(0),
    )

    assert plan.layer_placements == (
        LayerExpertPlacement(3, (0, 2), ()),
        LayerExpertPlacement(4, (1,), (3,)),
    )


def test_residency_profile_cap_leaves_extra_hbm_free(tmp_path, monkeypatch):
    path = tmp_path / "placement.json"
    write_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)
    monkeypatch.setenv("VLLM_TIERED_MOE_PROFILE_CAP", "1")

    plan = plan_rank_expert_tiers(
        manifest,
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=750,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"fixed": 300},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
        owned_expert_ids_by_layer=profile.ownership_for_rank(0),
        hot_expert_ids_by_layer=profile.hot_for_rank(0),
    )

    assert plan.layer_placements == (
        LayerExpertPlacement(3, (2,), (0,)),
        LayerExpertPlacement(4, (1,), (3,)),
    )


def test_residency_profile_demotes_hot_when_hbm_budget_shrinks(tmp_path):
    """A smaller HBM budget (e.g. concurrent-sequence KV growth) trims hot
    experts deterministically instead of failing."""
    path = tmp_path / "placement.json"
    write_placement_profile(path)
    manifest = make_planner_manifest()
    profile = load_tiered_moe_placement_profile(path, manifest, ep_size=2)

    plan = plan_rank_expert_tiers(
        manifest,
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=550,
        hbm_reserve_bytes=100,
        fixed_hbm_allocations={"fixed": 300},
        host_capacity_bytes=1000,
        host_reserve_bytes=100,
        minimum_hbm_reserve_bytes=0,
        minimum_host_reserve_bytes=0,
        owned_expert_ids_by_layer=profile.ownership_for_rank(0),
        hot_expert_ids_by_layer=profile.hot_for_rank(0),
    )

    assert plan.layer_placements == (
        LayerExpertPlacement(3, (), (0, 2)),
        LayerExpertPlacement(4, (1,), (3,)),
    )


def test_trace_placement_profile_rejects_unbalanced_owner_row(tmp_path):
    path = tmp_path / "placement.json"
    write_placement_profile(path)
    data = json.loads(path.read_text())
    data["owners"][0] = [0, 0, 0, 1]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="EP balanced"):
        load_tiered_moe_placement_profile(path, make_planner_manifest(), ep_size=2)


def test_planner_fails_when_cold_experts_exceed_grace_capacity():
    with pytest.raises(ValueError, match="paired Grace capacity"):
        plan_rank_expert_tiers(
            make_planner_manifest(),
            ep_size=2,
            ep_rank=0,
            hbm_capacity_bytes=650,
            hbm_reserve_bytes=100,
            fixed_hbm_allocations={"fixed": 300},
            host_capacity_bytes=250,
            host_reserve_bytes=100,
            minimum_hbm_reserve_bytes=0,
            minimum_host_reserve_bytes=0,
        )


def test_planner_fails_when_conversion_scratch_exceeds_hbm():
    with pytest.raises(ValueError, match="conversion scratch"):
        plan_rank_expert_tiers(
            make_planner_manifest(),
            ep_size=2,
            ep_rank=0,
            hbm_capacity_bytes=650,
            hbm_reserve_bytes=100,
            fixed_hbm_allocations={"fixed": 300},
            host_capacity_bytes=1000,
            host_reserve_bytes=100,
            transient_hbm_bytes=200,
            minimum_hbm_reserve_bytes=0,
            minimum_host_reserve_bytes=0,
        )


def test_plan_cli_requires_named_exact_bytes():
    assert _fixed_allocation("cache=1024") == ("cache", 1024)
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=BYTES"):
        _fixed_allocation("cache")


def test_glm_kv_plan_budgets_scheduler_null_block_for_400k():
    plan = plan_glm_kv_cache(
        make_glm_config(),
        max_model_len=400_000,
        block_size=64,
        kv_cache_dtype="fp8_ds_mla",
        main_cache_tier="host_uva",
    )

    assert plan.num_blocks == 6_251
    assert plan.allocated_tokens == 400_064
    assert plan.main_page_bytes_per_layer == 41_984
    assert plan.indexer_page_bytes_per_layer == 8_448
    assert plan.main_cache_bytes == 20_470_474_752
    assert plan.indexer_cache_bytes == 1_108_977_408
    assert plan.hbm_bytes == 1_108_977_408
    assert plan.host_bytes == 20_470_474_752


def test_glm_kv_plan_shards_blocks_across_dcp_group():
    """DCP4 stores 1/4 of every sequence per rank: one 64-token physical
    block per rank per 256-token logical block, plus the null block."""
    plan = plan_glm_kv_cache(
        make_glm_config(),
        max_model_len=400_000,
        block_size=64,
        kv_cache_dtype="fp8_ds_mla",
        main_cache_tier="hbm",
        dcp_world_size=4,
    )

    assert plan.dcp_world_size == 4
    assert plan.num_blocks == 1_564
    assert plan.allocated_tokens == 100_096
    assert (plan.num_blocks - 1) * plan.block_size * 4 >= 400_000
    assert plan.main_cache_bytes == 1_564 * 41_984 * 78
    assert plan.indexer_cache_bytes == 1_564 * 8_448 * 21


def test_glm_kv_plan_provisions_concurrent_sequences_under_dcp():
    """c=4 x 400K with DCP4 must land at the same per-rank footprint as
    c=1 x 400K without DCP (one full-length shard run per sequence)."""
    plan = plan_glm_kv_cache(
        make_glm_config(),
        max_model_len=400_000,
        block_size=64,
        kv_cache_dtype="fp8_ds_mla",
        main_cache_tier="hbm",
        dcp_world_size=4,
        max_num_seqs=4,
    )

    assert plan.num_blocks == 4 * 1_563 + 1
    assert plan.main_cache_bytes == (4 * 1_563 + 1) * 41_984 * 78


def test_glm_kv_plan_rejects_non_positive_dcp():
    with pytest.raises(ValueError, match="DCP world size"):
        plan_glm_kv_cache(
            make_glm_config(),
            max_model_len=400_000,
            block_size=64,
            kv_cache_dtype="fp8_ds_mla",
            main_cache_tier="hbm",
            dcp_world_size=0,
        )


def test_glm_kv_plan_includes_grafted_mtp_cache():
    config = make_glm_config()
    config["num_nextn_predict_layers"] = 1

    plan = plan_glm_kv_cache(
        config,
        max_model_len=400_000,
        block_size=64,
        kv_cache_dtype="fp8_ds_mla",
        main_cache_tier="hbm",
        num_mtp_layers=1,
    )

    assert plan.main_layer_count == 79
    assert len(plan.indexer_layer_ids) == 22
    assert plan.main_cache_bytes == 20_732_916_736
    assert plan.indexer_cache_bytes == 1_161_785_856


def test_glm_kv_plan_rejects_non_native_layout():
    config = make_glm_config()
    config["model_version"] = "deepseek_v4"

    with pytest.raises(ValueError, match="DeepSeek-v4"):
        plan_glm_kv_cache(
            config,
            max_model_len=400_000,
            block_size=64,
            kv_cache_dtype="fp8_ds_mla",
            main_cache_tier="hbm",
        )


def test_glm_runtime_buffer_plan_matches_two_marlin_tiers():
    manifest = TieredMoECheckpointManifest(
        model_path=Path("/model"),
        config_sha256="config",
        index_sha256="index",
        entries=(),
        checkpoint_bytes=0,
        routed_expert_bytes=0,
        non_routed_bytes=0,
        routed_layers=tuple(range(3, 78)),
        num_experts=256,
        checkpoint_expert_bytes=19_464_240,
        runtime_expert_bytes=19_464_200,
    )

    buffers = plan_tiered_glm_runtime_buffers(
        manifest,
        max_num_batched_tokens=8192,
        ep_size=4,
        sm_count=132,
    )

    assert buffers.intermediate_bytes == 3_221_225_472
    assert buffers.alignment_bytes == 663_528
    assert buffers.marlin_lock_bytes == 4_224
    assert buffers.placement_map_bytes == 249_600
    assert buffers.remap_bytes == 1_048_576
    assert buffers.steady_hbm_bytes == 3_223_191_400
    assert buffers.conversion_scratch_bytes == 67_108_864


def test_glm_marlin_destination_layout_is_exact_and_non_overlapping():
    buffer = torch.empty(2 * GLM_MARLIN_EXPERT_BYTES, dtype=torch.uint8)

    views = build_expert_component_views(buffer, expert_count=2)

    assert GLM_MARLIN_EXPERT_BYTES == 19_464_200
    assert set(views) == {component.name for component in GLM_MARLIN_COMPONENTS}
    intervals = []
    for component in GLM_MARLIN_COMPONENTS:
        view = views[component.name]
        assert view.shape == (2, *component.shape)
        assert view.dtype == component.dtype
        start = view.data_ptr()
        intervals.append((start, start + view.numel() * view.element_size()))
    assert intervals == sorted(intervals)
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert intervals[0][0] == buffer.data_ptr()
    assert intervals[-1][1] == buffer.data_ptr() + buffer.numel()


def test_auto_round_marlin_destination_layout_has_group64_scales():
    components = glm_marlin_components(64)
    expert_bytes = sum(component.bytes_per_expert for component in components)
    buffer = torch.empty(expert_bytes, dtype=torch.uint8)

    views = build_expert_component_views(buffer, expert_count=1, group_size=64)

    assert expert_bytes == 20_054_024
    assert views["w13_weight_scale"].shape == (1, 96, 4096)
    assert views["w2_weight_scale"].shape == (1, 32, 6144)


def test_final_destination_commits_exactly_one_complete_expert():
    buffer = torch.zeros(2 * GLM_MARLIN_EXPERT_BYTES, dtype=torch.uint8)
    views = build_expert_component_views(buffer, expert_count=2)
    tier = ExpertTierStorage((7, 9), buffer, views)
    converted = {
        component.name: torch.full(component.shape, 1, dtype=component.dtype)
        for component in GLM_MARLIN_COMPONENTS
    }

    tier.copy_expert_from(9, converted)

    for component in GLM_MARLIN_COMPONENTS:
        assert torch.count_nonzero(views[component.name][0]) == 0
        assert torch.all(views[component.name][1] == 1)
    with pytest.raises(ValueError, match="not assigned"):
        tier.copy_expert_from(8, converted)


def test_one_expert_stager_is_complete_and_bounded():
    stager = OneExpertCheckpointStager()
    shapes = {
        "gate_proj": {
            "weight_packed": ((2048, 768), torch.int32),
            "weight_scale": ((2048, 48), torch.bfloat16),
            "weight_shape": ((2,), torch.int64),
        },
        "up_proj": {
            "weight_packed": ((2048, 768), torch.int32),
            "weight_scale": ((2048, 48), torch.bfloat16),
            "weight_shape": ((2,), torch.int64),
        },
        "down_proj": {
            "weight_packed": ((6144, 256), torch.int32),
            "weight_scale": ((6144, 16), torch.bfloat16),
            "weight_shape": ((2,), torch.int64),
        },
    }
    bundle = None
    for projection, components in shapes.items():
        for component, (shape, dtype) in components.items():
            name = f"model.layers.3.mlp.experts.7.{projection}.{component}"
            bundle = stager.add(name, torch.empty(shape, dtype=dtype))

    assert bundle is not None
    assert bundle.layer_id == 3
    assert bundle.expert_id == 7
    assert bundle.num_bytes == 19_464_240
    stager.finish()


def test_one_auto_round_expert_stager_is_complete_and_bounded():
    stager = OneExpertCheckpointStager()
    shapes = {
        "gate_proj": {
            "qweight": ((768, 2048), torch.int32),
            "qzeros": ((96, 256), torch.int32),
            "scales": ((96, 2048), torch.float16),
        },
        "up_proj": {
            "qweight": ((768, 2048), torch.int32),
            "qzeros": ((96, 256), torch.int32),
            "scales": ((96, 2048), torch.float16),
        },
        "down_proj": {
            "qweight": ((256, 6144), torch.int32),
            "qzeros": ((32, 768), torch.int32),
            "scales": ((32, 6144), torch.float16),
        },
    }
    bundle = None
    for projection, components in shapes.items():
        for component, (shape, dtype) in components.items():
            name = f"model.layers.3.mlp.experts.7.{projection}.{component}"
            bundle = stager.add(name, torch.empty(shape, dtype=dtype))

    assert bundle is not None
    assert bundle.num_bytes == 20_348_928
    stager.finish()


def test_one_expert_stager_rejects_interleaved_experts():
    stager = OneExpertCheckpointStager()
    prefix = "model.layers.3.mlp.experts"
    stager.add(
        f"{prefix}.0.gate_proj.weight_shape",
        torch.empty(2, dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="interleaved or incomplete"):
        stager.add(
            f"{prefix}.1.gate_proj.weight_shape",
            torch.empty(2, dtype=torch.int64),
        )


def test_tiered_stream_passes_dense_weights_and_commits_one_expert(monkeypatch):
    class RecordingTier:
        expert_ids = (7,)

        def __init__(self):
            self.commits = []

        def copy_expert_from(self, expert_id, components):
            self.commits.append((expert_id, components))

    layer = torch.nn.Module()
    layer.quant_method = object()
    tier = RecordingTier()
    layer.tiered_moe_storage = LayerTieredExpertStorage(3, tier, None)
    model = torch.nn.Module()
    model.add_module("experts", layer)
    marker = {"converted": torch.tensor(1)}
    monkeypatch.setattr(
        streaming_module,
        "convert_glm_w4a16_expert",
        lambda bundle, layer, quant_method, device: marker,
    )
    shapes: dict[str, dict[str, tuple[tuple[int, ...], torch.dtype]]] = {
        "weight_packed": {
            "gate_proj": ((2048, 768), torch.int32),
            "up_proj": ((2048, 768), torch.int32),
            "down_proj": ((6144, 256), torch.int32),
        },
        "weight_scale": {
            "gate_proj": ((2048, 48), torch.bfloat16),
            "up_proj": ((2048, 48), torch.bfloat16),
            "down_proj": ((6144, 16), torch.bfloat16),
        },
        "weight_shape": {
            "gate_proj": ((2,), torch.int64),
            "up_proj": ((2,), torch.int64),
            "down_proj": ((2,), torch.int64),
        },
    }

    def weights():
        yield "model.embed_tokens.weight", torch.tensor(2)
        for component, projections in shapes.items():
            for projection, (shape, dtype) in projections.items():
                name = f"model.layers.3.mlp.experts.7.{projection}.{component}"
                yield name, torch.empty(shape, dtype=dtype)
        yield "model.norm.weight", torch.tensor(3)

    loader = TieredMoEExpertLoader(model, torch.device("cpu"))
    passed = list(loader.filter(weights()))
    loader.finish()

    assert [name for name, _ in passed] == [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    assert tier.commits == [(7, marker)]


def test_tiered_stream_rejects_missing_planned_expert():
    tier = SimpleNamespace(expert_ids=(7,), copy_expert_from=lambda *args: None)
    layer = torch.nn.Module()
    layer.quant_method = object()
    layer.tiered_moe_storage = LayerTieredExpertStorage(3, tier, None)
    model = torch.nn.Module()
    model.add_module("experts", layer)
    loader = TieredMoEExpertLoader(model, torch.device("cpu"))

    with pytest.raises(ValueError, match="1 missing experts"):
        loader.finish()


def test_tiered_marlin_dispatches_once_for_all_tiers():
    class ModularExperts:
        @staticmethod
        def is_monolithic():
            return False

    class Kernel:
        def __init__(self, value):
            self.value = value
            self.calls = []

        def apply_tiered(self, x, tiers, weights, ids, **kwargs):
            self.calls.append((tiers, kwargs))
            return torch.full_like(x, self.value)

    hot_kernel = Kernel(2)
    cold_kernel = Kernel(3)
    method = object.__new__(CompressedTensorsWNA16MarlinMoEMethod)
    method.moe_kernel = None
    method.experts_cls = ModularExperts
    method.tiered_moe_kernels = [
        (
            hot_kernel,
            {"w13_weight_packed": torch.empty(0), "w2_weight_packed": torch.empty(0)},
            torch.tensor([0, -1]),
        ),
        (
            cold_kernel,
            {"w13_weight_packed": torch.empty(0), "w2_weight_packed": torch.empty(0)},
            torch.tensor([-1, 0]),
        ),
    ]
    layer = SimpleNamespace(
        activation="silu",
        global_num_experts=2,
        expert_map=torch.tensor([0, 1]),
        apply_router_weight_on_input=False,
    )
    shared = object()

    output = method.apply(
        layer,
        torch.zeros(1, 4),
        torch.ones(1, 2),
        torch.tensor([[0, 1]]),
        shared,
        torch.zeros(1, 4),
    )

    assert torch.equal(output, torch.full((1, 4), 2.0))
    assert len(hot_kernel.calls) == 1
    tiers, kwargs = hot_kernel.calls[0]
    assert [tier[0] for tier in tiers] == [hot_kernel, cold_kernel]
    assert kwargs["prepare_expert_map"] is layer.expert_map
    assert kwargs["shared_experts"] is shared
    assert cold_kernel.calls == []


def test_tiered_modular_kernel_prepares_and_finalizes_once(monkeypatch):
    monkeypatch.setattr(mk, "aux_stream", lambda: None)
    calls = []

    def make_kernel(value, primary=False):
        impl = object.__new__(mk.FusedMoEKernelModularImpl)

        def prepare(*args):
            if not primary:
                raise AssertionError("A secondary tier must not prepare")
            calls.append("prepare")
            return args[0], None, None, args[2], args[1]

        def fused_experts(**kwargs):
            calls.append(f"experts-{value}")
            return torch.full_like(kwargs["a1q"], value)

        def finalize(output, fused_out, *args, **kwargs):
            if not primary:
                raise AssertionError("A secondary tier must not finalize")
            calls.append("finalize")
            output.copy_(fused_out)
            return output

        impl._prepare = prepare
        impl._fused_experts = fused_experts
        impl._finalize = finalize
        kernel = object.__new__(mk.FusedMoEKernel)
        kernel.impl = impl
        return kernel

    hot_kernel = make_kernel(2, primary=True)
    cold_kernel = make_kernel(3)
    output = hot_kernel.apply_tiered(
        hidden_states=torch.zeros(1, 4),
        tiers=[
            (
                hot_kernel,
                torch.empty(1, 0, 0),
                torch.empty(1, 0, 0),
                torch.tensor([0, -1]),
            ),
            (
                cold_kernel,
                torch.empty(1, 0, 0),
                torch.empty(1, 0, 0),
                torch.tensor([-1, 0]),
            ),
        ],
        topk_weights=torch.ones(1, 2),
        topk_ids=torch.tensor([[0, 1]]),
        activation=mk.MoEActivation.SILU,
        global_num_experts=2,
        prepare_expert_map=torch.tensor([0, 1]),
        apply_router_weight_on_input=False,
    )

    assert torch.equal(output, torch.full((1, 4), 5.0))
    assert calls == ["prepare", "experts-2", "experts-3", "finalize"]


def test_tiered_modular_kernel_allocates_independent_workspaces(monkeypatch):
    class Experts:
        @staticmethod
        def workspace_dtype(out_dtype):
            return out_dtype

        @staticmethod
        def workspace_shapes(M, *args):
            return (M, 4), (M, 2), (M, 4)

        @staticmethod
        def moe_problem_size(a1q, *args):
            return 0, a1q.shape[0], 4, 4, 2

    class WorkspaceManager:
        def __init__(self):
            self.requests = ()

        def get_simultaneous(self, *requests):
            self.requests = requests
            return [torch.empty(shape, dtype=dtype) for shape, dtype in requests]

    manager = WorkspaceManager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: manager)
    tiers = []
    for num_experts in (2, 3):
        impl = object.__new__(mk.FusedMoEKernelModularImpl)
        impl.fused_experts = Experts()
        kernel = object.__new__(mk.FusedMoEKernel)
        kernel.impl = impl
        tiers.append(
            (
                kernel,
                torch.empty(num_experts, 1, 1),
                torch.empty(num_experts, 1, 1),
                None,
            )
        )

    buffers = tiers[0][0].impl._allocate_tiered_buffers(
        torch.float32,
        torch.empty(1, 4),
        torch.tensor([[0, 1]]),
        mk.MoEActivation.SILU,
        2,
        None,
        tiers,
    )

    assert len(manager.requests) == 4
    assert buffers[0][0].data_ptr() == buffers[0][2].data_ptr()
    assert buffers[1][0].data_ptr() == buffers[1][2].data_ptr()
    assert buffers[0][0].data_ptr() != buffers[1][0].data_ptr()
    assert buffers[0][1].data_ptr() != buffers[1][1].data_ptr()


def test_rank_load_plan_context_is_scoped():
    plan = object()
    assert get_tiered_moe_rank_load_plan() is None

    with use_tiered_moe_rank_load_plan(plan):  # type: ignore[arg-type]
        assert get_tiered_moe_rank_load_plan() is plan

    assert get_tiered_moe_rank_load_plan() is None


def test_rank_load_plan_resolves_layer_prefix():
    plan = SimpleNamespace(
        rank_plan=SimpleNamespace(
            layer_placements=(
                LayerExpertPlacement(3, (0,), (1,)),
                LayerExpertPlacement(4, (1,), (0,)),
            )
        )
    )

    placement = resolve_layer_expert_placement(
        plan,
        "model.layers.4.mlp.experts",  # type: ignore[arg-type]
    )

    assert placement.layer_id == 4
    with pytest.raises(ValueError, match="no placement"):
        resolve_layer_expert_placement(
            plan,
            "model.layers.5.mlp.experts",  # type: ignore[arg-type]
        )


def test_machine_profile_is_hashed_and_validated(tmp_path):
    profile_path = tmp_path / "gh200.json"
    profile = {
        "profile_version": 1,
        "machine": "GH200",
        "hbm_capacity_bytes": 100,
        "host_capacity_bytes": 200,
        "gpu_sm_count": 132,
        "fixed_hbm_allocations": {"runtime": 10},
        "fixed_host_allocations": {},
        "measurement_source": "test fixture",
    }
    profile_path.write_text(json.dumps(profile))

    loaded = load_grace_machine_profile(profile_path)

    assert loaded.hbm_capacity_bytes == 100
    assert dict(loaded.fixed_hbm_allocations) == {"runtime": 10}
    assert len(loaded.sha256) == 64

    profile["unexpected"] = True
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="fields do not match"):
        load_grace_machine_profile(profile_path)


def test_planner_accounts_for_fixed_host_bytes_and_minimum_reserve():
    with pytest.raises(ValueError, match="Host reserve"):
        plan_rank_expert_tiers(
            make_planner_manifest(),
            ep_size=2,
            ep_rank=0,
            hbm_capacity_bytes=10_000_000_000,
            hbm_reserve_bytes=7_000_000_000,
            fixed_hbm_allocations={"weights": 100},
            host_capacity_bytes=20_000_000_000,
            host_reserve_bytes=7_999_999_999,
            fixed_host_allocations={"main_mla_cache": 100},
        )

    plan = plan_rank_expert_tiers(
        make_planner_manifest(),
        ep_size=2,
        ep_rank=0,
        hbm_capacity_bytes=10_000_000_000,
        hbm_reserve_bytes=7_000_000_000,
        fixed_hbm_allocations={"weights": 100},
        host_capacity_bytes=20_000_000_000,
        host_reserve_bytes=8_000_000_000,
        fixed_host_allocations={"main_mla_cache": 1_000},
    )

    assert plan.fixed_host_bytes == 1_000
    assert plan.planned_host_bytes == 8_000_001_000


def test_observed_hbm_reserve_fails_closed_below_runtime_margin():
    config = SimpleNamespace(
        tiered_moe_config=SimpleNamespace(enabled=True, hbm_reserve_gb=7.0)
    )

    assert validate_tiered_moe_observed_hbm_reserve(config, 6_000_000_000) == (
        6_000_000_000
    )
    with pytest.raises(RuntimeError, match="Replan more experts"):
        validate_tiered_moe_observed_hbm_reserve(config, 5_999_999_999)

    low_reserve_config = SimpleNamespace(
        tiered_moe_config=SimpleNamespace(enabled=True, hbm_reserve_gb=5.0)
    )
    assert (
        validate_tiered_moe_observed_hbm_reserve(low_reserve_config, 4_000_000_000)
        == 4_000_000_000
    )
    with pytest.raises(RuntimeError, match="Replan more experts"):
        validate_tiered_moe_observed_hbm_reserve(low_reserve_config, 3_999_999_999)


def make_entry(name: str, num_bytes: int) -> TensorManifestEntry:
    return TensorManifestEntry(
        name=name,
        shard="model.safetensors",
        dtype="U8",
        shape=(num_bytes,),
        num_bytes=num_bytes,
    )


def test_non_routed_inventory_accounts_for_tp_drop_and_fusion():
    entries = (
        make_entry("model.embed_tokens.weight", 400),
        make_entry("model.norm.weight", 16),
        make_entry("model.layers.0.self_attn.q_b_proj.weight_packed", 400),
        make_entry("model.layers.0.self_attn.q_b_proj.weight_shape", 16),
        make_entry("model.layers.3.self_attn.indexer.wq_b.weight", 200),
        make_entry("model.layers.0.self_attn.q_a_proj.weight_shape", 16),
        make_entry("model.layers.0.self_attn.kv_a_proj_with_mqa.weight_shape", 16),
        make_entry("model.layers.78.eh_proj.weight", 64),
        make_entry("model.layers.78.mlp.experts.0.gate_proj.weight", 64),
    )
    checkpoint_bytes = sum(entry.num_bytes for entry in entries)
    manifest = TieredMoECheckpointManifest(
        model_path=Path("/model"),
        config_sha256="config",
        index_sha256="index",
        entries=entries,
        checkpoint_bytes=checkpoint_bytes,
        routed_expert_bytes=0,
        non_routed_bytes=checkpoint_bytes,
        routed_layers=(3,),
        num_experts=1,
        checkpoint_expert_bytes=100,
        runtime_expert_bytes=100,
    )

    inventory = build_glm_non_routed_runtime_inventory(manifest, tp_size=4)

    assert inventory.tp_sharded_checkpoint_bytes == 800
    assert inventory.tp_sharded_runtime_bytes == 200
    assert inventory.ep_sharded_checkpoint_bytes == 64
    assert inventory.ep_sharded_runtime_bytes == 16
    assert inventory.replicated_bytes == 128
    assert inventory.dropped_checkpoint_bytes == 200
    assert inventory.fusion_savings_bytes == 16
    assert inventory.format_conversion_bytes == 0
    assert inventory.runtime_bytes_per_rank == 328
