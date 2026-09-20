"""Conservative inference-memory estimates without importing a tensor framework.

These are planning estimates, not measured peaks or guarantees against OOM. The
caller supplies resident parameter bytes, not the checkpoint's compressed size.
Apple preparation must also check the separate CPU reference/conversion phase.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIB = 1024**2
GIB = 1024**3


class HardwareMetrics(BaseModel):
    """Byte counts; GPU metrics belong to the named *logical* CUDA device."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    host_total_bytes: Optional[int] = Field(default=None, ge=0)
    host_available_bytes: Optional[int] = Field(default=None, ge=0)
    gpu_total_bytes: Optional[int] = Field(default=None, ge=0)
    gpu_available_bytes: Optional[int] = Field(default=None, ge=0)
    gpu_device: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_memory(self):
        for pool in ("host", "gpu"):
            total = getattr(self, f"{pool}_total_bytes")
            available = getattr(self, f"{pool}_available_bytes")
            if total is not None and available is not None and available > total:
                raise ValueError(f"{pool} available memory exceeds total memory")
        return self


class ResourcePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1
    parameter_bytes: int
    checkpoint_storage_bytes: Optional[int]
    device: str
    backend: Literal["torch", "mlx"]
    batch_size: int
    sequence_length: int
    memory_pool: Literal["host", "apple_shared", "gpu"]
    minimum_required_bytes: int
    estimated_peak_bytes: int
    workspace_bytes: int
    available_bytes: Optional[int]
    total_bytes: Optional[int]
    host_minimum_required_bytes: int
    host_estimated_peak_bytes: int
    host_available_bytes: Optional[int]
    host_total_bytes: Optional[int]
    gpu_minimum_required_bytes: Optional[int]
    gpu_estimated_peak_bytes: Optional[int]
    gpu_available_bytes: Optional[int]
    gpu_total_bytes: Optional[int]
    risk: Literal["within_budget", "memory_pressure", "over_budget", "unknown"]
    blocked: bool
    allow_memory_risk: bool
    warnings: list[str]
    suggestions: list[str]
    assumptions: list[str]
    sources: list[str]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parameter_bytes: int = Field(gt=0)
    checkpoint_storage_bytes: Optional[int] = Field(default=None, gt=0)
    device: str
    backend: Literal["torch", "mlx"]
    batch_size: int = Field(ge=1, le=4096)
    sequence_length: int = Field(ge=1, le=32768)
    allow_memory_risk: bool


def _device(value: str) -> str:
    value = value.strip().lower()
    if value in ("gpu", "cuda"):
        return "cuda:0"
    if value in ("cpu", "apple"):
        return value
    if re.fullmatch(r"cuda:[0-9]+", value):
        return f"cuda:{int(value.split(':')[1])}"
    raise ValueError("device must be cpu, apple, gpu, cuda, or cuda:N")


def _linux_memory(text: str) -> tuple[Optional[int], Optional[int]]:
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(MemTotal|MemAvailable|MemFree):\s+(\d+)\s+kB\s*", line)
        if match:
            values[match[1]] = int(match[2]) * 1024
    # MemFree alone omits reclaimable pages but is a safe older-kernel fallback.
    return values.get("MemTotal"), values.get("MemAvailable", values.get("MemFree"))


def _mac_available(text: str) -> Optional[int]:
    page_size = re.search(r"page size of (\d+) bytes", text)
    if not page_size or int(page_size[1]) <= 0:
        return None
    counts = {}
    for line in text.splitlines():
        match = re.fullmatch(r"Pages (free|inactive|speculative):\s+(\d+)\.\s*", line)
        if match:
            counts[match[1]] = int(match[2])
    if set(counts) != {"free", "inactive", "speculative"}:
        return None
    # Purgeable is not added: it overlaps other VM categories. Compressed pages
    # are also excluded; pressure and the OS may reclaim less than this estimate.
    return sum(counts.values()) * int(page_size[1])


def _command(argv: list[str]) -> str:
    result = subprocess.run(argv, check=True, text=True, capture_output=True, timeout=3)
    if len(result.stdout) > 1024 * 1024:
        raise ValueError("Unexpectedly large hardware query output")
    return result.stdout


def _nvidia_memory(device: str, environment: Mapping[str, str]) -> tuple[Optional[int], Optional[int], str]:
    """Never equate nvidia-smi's physical index with CUDA's logical index."""
    output = _command(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free,mig.mode.current",
                       "--format=csv,noheader,nounits"])
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 4:
            raise ValueError("Unrecognized NVIDIA memory output")
        uuid, total, free, mig_mode = (part.strip() for part in row)
        if not uuid.startswith("GPU-") or not total.isdigit() or not free.isdigit():
            raise ValueError("NVIDIA device does not expose ordinary GPU memory metrics")
        if int(free) > int(total):
            raise ValueError("NVIDIA free memory exceeds total memory")
        rows.append((uuid, int(total) * MIB, int(free) * MIB, mig_mode))
    index = int(device.split(":")[1])
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    selected = None
    if visible is not None:
        identities = [item.strip() for item in visible.split(",")]
        # UUID-only lists have explicit mapping. Numeric lists may refer to a
        # different CUDA ordering, and MIG slices have different memory pools.
        if index < len(identities) and all(item.startswith("GPU-") for item in identities):
            if len(set(identities)) == len(identities):
                selected = next((row for row in rows if row[0] == identities[index]), None)
        elif len(rows) == 1 and index == 0 and identities == ["0"]:
            selected = rows[0]
    elif len(rows) == 1 and index == 0:
        selected = rows[0]
    if selected is None:
        return None, None, ("GPU memory is unknown: nvidia-smi cannot safely map the selected logical "
                            "CUDA device. Query memory with Torch in the selected runtime environment.")
    if selected[3].strip("[]").lower() not in ("disabled", "n/a", "na"):
        return None, None, ("GPU memory is unknown: MIG mode is enabled or unrecognized; a physical "
                            "GPU's capacity is not a CUDA partition's capacity. Query the selected Torch runtime.")
    return selected[1], selected[2], ""


def detect_hardware(device: str = "cpu", *, environment: Optional[Mapping[str, str]] = None) -> HardwareMetrics:
    """Read cheap local OS metrics; failed/ambiguous queries remain unknown."""
    selected = _device(device)
    total = available = gpu_total = gpu_available = None
    sources, warnings = [], []
    try:
        system = platform.system()
        if system == "Linux":
            total, available = _linux_memory(Path("/proc/meminfo").read_text())
            sources.append("/proc/meminfo (host metrics; container limits may be lower)")
        elif system == "Darwin":
            total = int(_command(["sysctl", "-n", "hw.memsize"]).strip())
            available = _mac_available(_command(["vm_stat"]))
            sources.append("sysctl hw.memsize; vm_stat free+inactive+speculative pages")
        else:
            warnings.append(f"Host memory detection is not implemented for {system}.")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        warnings.append(f"Host memory query unavailable ({type(exc).__name__}).")
    if total is not None and total <= 0:
        total = None
    if total is not None and available is not None:
        available = min(available, total)
    if selected.startswith("cuda:"):
        try:
            gpu_total, gpu_available, warning = _nvidia_memory(
                selected, os.environ if environment is None else environment)
            if warning:
                warnings.append(warning)
            else:
                sources.append("nvidia-smi UUID-mapped device memory")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            warnings.append(f"GPU memory query unavailable ({type(exc).__name__}); use the selected Torch runtime.")
    return HardwareMetrics(host_total_bytes=total, host_available_bytes=available,
                           gpu_total_bytes=gpu_total, gpu_available_bytes=gpu_available,
                           gpu_device=selected if gpu_total is not None else None,
                           sources=sources, warnings=warnings)


def plan_resources(parameter_bytes: int, device: str, backend: str = "torch", batch_size: int = 1,
                   sequence_length: int = 1024, allow_memory_risk: bool = False,
                   hardware: Optional[Union[HardwareMetrics, dict]] = None,
                   checkpoint_storage_bytes: Optional[int] = None) -> ResourcePlan:
    """Estimate peak resident memory for one selected backend; never change it.

    CPU: 2.5x parameter bytes + workspace; Apple: 1.25x + workspace;
    CUDA: 1.25x + workspace in VRAM, plus 2x weights + 256 MiB host staging.
    Workspace reserves 256 MiB plus 512 KiB per batched token and 128 bytes
    per batched attention token pair, deliberately covering unfused attention
    and temporary activations without needing architecture/framework imports.
    If supplied, checkpoint storage includes every unique serialized tensor
    storage, including optimizer state; loading it may temporarily coexist with
    model parameters on CPU/CUDA. Apple bundle inference ignores this extra
    loading allocation; its CPU preparation phase must be planned separately.
    """
    request = _Request(parameter_bytes=parameter_bytes, device=device, backend=backend,
                       batch_size=batch_size, sequence_length=sequence_length,
                       allow_memory_risk=allow_memory_risk,
                       checkpoint_storage_bytes=checkpoint_storage_bytes)
    selected = _device(request.device)
    metrics = (detect_hardware(selected) if hardware is None else
               hardware if isinstance(hardware, HardwareMetrics) else HardwareMetrics.model_validate(hardware))
    warnings = list(metrics.warnings)
    suggestions = []
    base = 256 * MIB
    workspace = base + batch_size * sequence_length * (512 * 1024 + sequence_length * 128)
    cpu_peak = (parameter_bytes * 5 + 1) // 2 + workspace
    minimum = parameter_bytes + base
    fast_peak = (parameter_bytes * 5 + 3) // 4 + workspace
    checkpoint_peak = (checkpoint_storage_bytes + parameter_bytes + workspace
                       if checkpoint_storage_bytes is not None else 0)
    cpu_peak = max(cpu_peak, checkpoint_peak)
    gpu = selected.startswith("cuda:")
    gpu_total = metrics.gpu_total_bytes if gpu else None
    gpu_available = metrics.gpu_available_bytes if gpu else None
    if gpu and (metrics.gpu_device is None or _device(metrics.gpu_device) != selected):
        gpu_total = gpu_available = None
        warnings.append("Supplied GPU memory has no matching logical CUDA device; VRAM remains unknown.")
    host_peak = parameter_bytes * 2 + base if gpu else fast_peak if selected == "apple" else cpu_peak
    if gpu and checkpoint_storage_bytes is not None:
        host_peak = max(host_peak, checkpoint_storage_bytes + parameter_bytes + base)
    host_minimum = parameter_bytes if gpu else minimum
    peak = max(fast_peak, checkpoint_peak) if gpu else host_peak
    available = gpu_available if gpu else metrics.host_available_bytes
    total = gpu_total if gpu else metrics.host_total_bytes
    pools = [("host", host_minimum, host_peak, metrics.host_available_bytes, metrics.host_total_bytes)]
    if gpu:
        pools.append(("GPU", minimum, peak, gpu_available, gpu_total))
    over, pressure, unknown = [], [], []
    for name, lower, estimate, free, capacity in pools:
        if capacity is not None and lower > capacity:
            over.append(name)
            warnings.append(f"{name} minimum estimated requirement exceeds total physical memory.")
        elif free is None and capacity is not None and estimate > capacity:
            over.append(name)
            warnings.append(f"{name} estimated peak exceeds total physical memory.")
        elif free is not None and estimate > free:
            over.append(name)
            warnings.append(f"{name} estimated peak exceeds currently available memory.")
        elif free is not None and estimate * 5 >= free * 4:
            pressure.append(name)
            warnings.append(f"{name} estimated peak uses at least 80% of currently available memory.")
        if free is None:
            unknown.append(name)
            warnings.append(f"{name} available memory is unknown; fit cannot be established.")
    risk = "over_budget" if over else "unknown" if unknown else "memory_pressure" if pressure else "within_budget"
    if over or pressure:
        if gpu and "GPU" in over and metrics.host_available_bytes is not None and cpu_peak <= metrics.host_available_bytes:
            suggestions.append("Use --device cpu; the CPU estimate fits host memory, but inference may be slower.")
        if batch_size > 1:
            suggestions.append("Reduce batch size to lower activation memory; rerun this estimate.")
        suggestions.append("Choose a smaller compatible model or free memory, then check again.")
    if over and allow_memory_risk:
        warnings.append("Memory-risk override is enabled; continuing may exhaust memory or cause severe swapping.")
    if selected == "cpu" and parameter_bytes >= 8 * GIB:
        warnings.append("Large-model CPU inference may be very slow; benchmark one representative batch before a full run.")
    assumptions = [
        "Estimates are conservative planning heuristics, not measured peaks or a guarantee against out-of-memory errors.",
        "parameter_bytes describes resident runtime weights; compressed checkpoints and optimizer tensors are excluded.",
        "When supplied, checkpoint_storage_bytes includes all unique tensor storages (including optimizer state); CPU/CUDA loading reserves these plus model parameters.",
        "Workspace = 256 MiB + batch*length*512 KiB + batch*length^2*128 bytes; architecture and kernels can change actual use.",
        "Memory can change after this snapshot; OS reservations, containers and other processes can reduce usable capacity.",
        "No model, precision, backend or device is changed automatically.",
    ]
    if selected == "apple":
        assumptions.append("Apple CPU and GPU share one physical memory pool; their capacities are not added. Plan CPU reference/conversion separately.")
    return ResourcePlan(parameter_bytes=parameter_bytes, checkpoint_storage_bytes=checkpoint_storage_bytes,
                        device=selected, backend=request.backend,
                        batch_size=batch_size, sequence_length=sequence_length,
                        memory_pool="gpu" if gpu else "apple_shared" if selected == "apple" else "host",
                        minimum_required_bytes=minimum, estimated_peak_bytes=peak, workspace_bytes=workspace,
                        available_bytes=available, total_bytes=total,
                        host_minimum_required_bytes=host_minimum, host_estimated_peak_bytes=host_peak,
                        host_available_bytes=metrics.host_available_bytes, host_total_bytes=metrics.host_total_bytes,
                        gpu_minimum_required_bytes=minimum if gpu else None,
                        gpu_estimated_peak_bytes=peak if gpu else None,
                        gpu_available_bytes=gpu_available, gpu_total_bytes=gpu_total,
                        risk=risk, blocked=bool(over) and not allow_memory_risk,
                        allow_memory_risk=allow_memory_risk, warnings=warnings, suggestions=suggestions,
                        assumptions=assumptions, sources=list(metrics.sources))
