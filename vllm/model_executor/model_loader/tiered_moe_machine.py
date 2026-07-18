# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated physical machine profiles for tiered MoE planning."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraceMachineProfile:
    """Exact per-rank capacities and measured non-model runtime allocations."""

    path: Path
    sha256: str
    profile_version: int
    machine: str
    hbm_capacity_bytes: int
    host_capacity_bytes: int
    gpu_sm_count: int
    fixed_hbm_allocations: tuple[tuple[str, int], ...]
    fixed_host_allocations: tuple[tuple[str, int], ...]
    measurement_source: str

    def summary(self) -> dict:
        """Return a JSON-compatible profile summary."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "profile_version": self.profile_version,
            "machine": self.machine,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "host_capacity_bytes": self.host_capacity_bytes,
            "gpu_sm_count": self.gpu_sm_count,
            "fixed_hbm_allocations": dict(self.fixed_hbm_allocations),
            "fixed_host_allocations": dict(self.fixed_host_allocations),
            "measurement_source": self.measurement_source,
        }


def _positive_int(data: dict[str, Any], name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Machine profile {name} must be a positive integer")
    return value


def _allocation_map(data: dict[str, Any], name: str) -> tuple[tuple[str, int], ...]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Machine profile {name} must be an object")
    allocations = []
    for allocation_name, num_bytes in value.items():
        if not isinstance(allocation_name, str) or not allocation_name:
            raise ValueError(f"Machine profile {name} has an invalid name")
        if (
            not isinstance(num_bytes, int)
            or isinstance(num_bytes, bool)
            or num_bytes < 0
        ):
            raise ValueError(
                f"Machine profile allocation {allocation_name} must be bytes"
            )
        allocations.append((allocation_name, num_bytes))
    return tuple(sorted(allocations))


def load_grace_machine_profile(path: str | Path) -> GraceMachineProfile:
    """Load a fail-closed machine profile without probing or allocating GPUs."""
    profile_path = Path(path).resolve()
    raw = profile_path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Machine profile must contain a JSON object")
    if set(data) != {
        "profile_version",
        "machine",
        "hbm_capacity_bytes",
        "host_capacity_bytes",
        "gpu_sm_count",
        "fixed_hbm_allocations",
        "fixed_host_allocations",
        "measurement_source",
    }:
        raise ValueError("Machine profile fields do not match schema version 1")
    profile_version = data["profile_version"]
    if profile_version != 1:
        raise ValueError("Only machine profile version 1 is supported")
    machine = data["machine"]
    measurement_source = data["measurement_source"]
    if not isinstance(machine, str) or not machine:
        raise ValueError("Machine profile machine must be a non-empty string")
    if not isinstance(measurement_source, str) or not measurement_source:
        raise ValueError("Machine profile measurement_source must be non-empty")
    return GraceMachineProfile(
        path=profile_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        profile_version=profile_version,
        machine=machine,
        hbm_capacity_bytes=_positive_int(data, "hbm_capacity_bytes"),
        host_capacity_bytes=_positive_int(data, "host_capacity_bytes"),
        gpu_sm_count=_positive_int(data, "gpu_sm_count"),
        fixed_hbm_allocations=_allocation_map(data, "fixed_hbm_allocations"),
        fixed_host_allocations=_allocation_map(data, "fixed_host_allocations"),
        measurement_source=measurement_source,
    )
