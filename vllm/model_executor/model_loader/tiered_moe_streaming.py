# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-expert streaming into tiered GLM Marlin destinations."""

import time
from collections.abc import Generator, Iterable

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.tiered_moe_conversion import (
    OneExpertCheckpointStager,
    convert_glm_w4a16_expert,
    is_glm_expert_checkpoint_tensor,
)
from vllm.model_executor.model_loader.tiered_moe_storage import (
    LayerTieredExpertStorage,
)

logger = init_logger(__name__)


class TieredMoEExpertLoader:
    """Consume routed weights one expert at a time and pass through the rest."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.device = device
        self.stager = OneExpertCheckpointStager()
        self.layers: dict[int, torch.nn.Module] = {}
        self.expected: set[tuple[int, int]] = set()
        self.loaded: set[tuple[int, int]] = set()
        self.conversion_seconds = 0.0
        for module in model.modules():
            storage = getattr(module, "tiered_moe_storage", None)
            if storage is None:
                continue
            if not isinstance(storage, LayerTieredExpertStorage):
                raise TypeError("Tiered MoE storage has an unexpected type")
            if storage.layer_id in self.layers:
                raise ValueError(f"Duplicate tiered routed layer {storage.layer_id}")
            self.layers[storage.layer_id] = module
            for tier in (storage.hot, storage.cold):
                if tier is not None:
                    self.expected.update(
                        (storage.layer_id, expert_id) for expert_id in tier.expert_ids
                    )
        if not self.layers:
            raise ValueError("Tiered MoE model has no compact expert destinations")

    def filter(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield non-routed tensors and commit complete routed experts."""
        for name, tensor in weights:
            if not is_glm_expert_checkpoint_tensor(name):
                yield name, tensor
                continue
            bundle = self.stager.add(name, tensor)
            if bundle is None:
                continue
            key = (bundle.layer_id, bundle.expert_id)
            if key not in self.expected:
                raise ValueError(f"Checkpoint expert {key} has no tiered destination")
            if key in self.loaded:
                raise ValueError(f"Checkpoint expert {key} was loaded twice")
            layer = self.layers[bundle.layer_id]
            quant_method = getattr(layer, "quant_method", None)
            if quant_method is None:
                raise ValueError(f"Tiered layer {bundle.layer_id} has no quant method")
            started = time.perf_counter()
            converted = convert_glm_w4a16_expert(
                bundle, layer, quant_method, self.device
            )
            storage = layer.tiered_moe_storage
            tier = self._resolve_tier(storage, bundle.expert_id)
            tier.copy_expert_from(bundle.expert_id, converted)
            self.conversion_seconds += time.perf_counter() - started
            self.loaded.add(key)

    @staticmethod
    def _resolve_tier(storage: LayerTieredExpertStorage, expert_id: int):
        for tier in (storage.hot, storage.cold):
            if tier is not None and expert_id in tier.expert_ids:
                return tier
        raise ValueError(
            f"Expert {expert_id} is absent from tiered layer {storage.layer_id}"
        )

    def finish(self) -> None:
        """Validate that every planned expert was committed exactly once."""
        self.stager.finish()
        missing = self.expected - self.loaded
        if missing:
            preview = ", ".join(str(key) for key in sorted(missing)[:8])
            raise ValueError(
                f"Tiered checkpoint ended with {len(missing)} missing experts: "
                f"{preview}"
            )
        logger.info_once(
            "Streamed %d routed experts into final tiers in %.2f seconds",
            len(self.loaded),
            self.conversion_seconds,
        )
