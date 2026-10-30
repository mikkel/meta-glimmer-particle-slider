"""Physical-device identity and owned-process helpers for the private studio."""


import os
from pathlib import Path
import re


GiB = 1024**3
HOST_RESERVE = 16 * GiB
VRAM_CAP = 30 * GiB


def canonical_uuid(value, *, require_prefix=False):
    text = str(value)
    if require_prefix and not text.startswith("GPU-"):
        raise ValueError("Expected a GPU-prefixed release UUID")
    if text.startswith("GPU-"):
        text = text[4:]
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", text):
        raise ValueError("Malformed complete GPU UUID")
    return "GPU-" + text.lower()


def process_memory(pid):
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
            values[key] = int(value.split()[0]) * 1024
    return values


def descendants(parent):
    records = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                records[int(entry.name)] = int(fields[1])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    owned = {int(parent)}
    while True:
        expanded = owned | {pid for pid, ppid in records.items() if ppid in owned}
        if expanded == owned:
            return sorted(owned)
        owned = expanded


def require_cuda_identity(expected_uuid):
    expected = canonical_uuid(expected_uuid, require_prefix=True)
    if canonical_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", ""), require_prefix=True) != expected:
        raise RuntimeError("CUDA visibility must contain exactly the released GPU UUID")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device required")
    properties = torch.cuda.get_device_properties(0)
    observed = canonical_uuid(properties.uuid)
    if observed != expected:
        raise RuntimeError(f"CUDA physical UUID mismatch: expected={expected}, observed={observed}")
    # Leave two GiB beneath the NVML cap for driver/library allocations outside
    # PyTorch's allocator. This affects only the owned process's visible device.
    torch.cuda.set_per_process_memory_fraction(min(1., (VRAM_CAP-2*GiB)/properties.total_memory), device=0)
    return observed
