# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for tiered MoE execution on coherent CPU-GPU memory."""

from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

TieredMoEBackend = Literal["auto", "uva", "cpu"]
MLACacheTier = Literal["auto", "hbm", "host_uva"]
ReplicaAssignment = Literal["off", "secondary", "greedy"]


@config
class TieredMoEConfig:
    """Configuration for static HBM/host-UVA MoE expert placement."""

    enabled: bool = False
    """Enable tiered MoE planning, loading, and execution."""

    backend: TieredMoEBackend = "auto"
    """Cold expert backend: auto, direct UVA, or the gated CPU contingency."""

    placement_profile: str | None = None
    """Optional static per-layer expert placement profile path."""

    replica_assignment: ReplicaAssignment = "off"
    """Replica assignment: primary-only, static secondary, or greedy decode."""

    routing_trace_output: str | None = None
    """Optional path for writing a sequence-level routing trace."""

    hbm_reserve_gb: float = Field(default=7.0, ge=0)
    """Planned decimal GB of HBM left outside physical allocations per rank."""

    host_reserve_gb: float = Field(default=8.0, ge=0)
    """Planned decimal GB of paired host memory left free per rank."""

    numa_strict: bool = True
    """Require at least 95 percent of audited host pages on the paired NUMA node."""

    plan_only: bool = False
    """Print the physical allocation plan and exit before reading model payloads."""

    mla_cache_tier: MLACacheTier = "auto"
    """Physical tier for the main MLA cache; the indexer cache remains in HBM."""

    grace_machine_profile: str | None = None
    """Optional measured Grace/Hopper topology and bandwidth profile path."""

    @model_validator(mode="after")
    def validate_tiered_moe_config(self) -> "TieredMoEConfig":
        """Enforce mandatory reserves whenever the tiered path is requested."""
        if self.plan_only and not self.enabled:
            raise ValueError("tiered_moe_plan_only requires enable_tiered_moe")
        if self.replica_assignment != "off" and not self.enabled:
            raise ValueError("tiered_moe_replica_assignment requires enable_tiered_moe")
        if self.replica_assignment != "off" and self.placement_profile is None:
            raise ValueError(
                "tiered_moe_replica_assignment requires a placement profile"
            )
        if self.enabled and self.hbm_reserve_gb < 5.0:
            raise ValueError("Tiered MoE requires at least 5 GB HBM reserve")
        if self.enabled and self.host_reserve_gb < 8.0:
            raise ValueError("Tiered MoE requires at least 8 GB host reserve")
        if self.enabled and self.grace_machine_profile is None:
            raise ValueError("Tiered MoE requires grace_machine_profile")
        if self.enabled and not self.plan_only and self.mla_cache_tier == "auto":
            raise ValueError("Tiered MoE loading requires an explicit MLA cache tier")
        return self

    def compute_hash(self) -> str:
        """Hash every field because placement changes the compiled graph."""
        from vllm.config.utils import get_hash_factors, hash_factors

        return hash_factors(get_hash_factors(self, ignored_factors=set()))
