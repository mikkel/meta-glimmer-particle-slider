"""Separate portable inference artifacts from complete training continuations."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .host import Slider, signature
from .upstream import REVISION

TEACHER = "glimmer-particle-teacher-v1"
STUDENT = "glimmer-ridge-lora-v1"


class OutputStore:
    def __init__(self, root, *, reserve_bytes=8 * 1024**3):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.reserve_bytes = int(reserve_bytes)

    @contextmanager
    def writing(self, name, estimated_bytes=0):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("Output name must be one relative filename")
        target = self.root / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        if shutil.disk_usage(self.root).free < self.reserve_bytes + estimated_bytes:
            raise OSError("Insufficient free output storage with reserve")
        fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=self.root)
        os.close(fd)
        try:
            yield Path(tmp)
            # Atomic publication, refuses to replace a concurrently created file.
            os.link(tmp, target)
        finally:
            os.unlink(tmp)


def validate_state(state, expected):
    if state.keys() != expected.keys():
        raise ValueError("Artifact tensor names differ")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"Artifact tensor shape/dtype differs: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite artifact tensor: {name}")
        if name.endswith(".alpha") and float(tensor) != 8:
            raise ValueError("Expected upstream alpha=rank=8")


def save_adapter(store, name, slider, *, state=None, kind=None, calibration=None, training=None):
    kind = kind or (STUDENT if slider.ordinary else TEACHER)
    if kind not in (TEACHER, STUDENT):
        raise ValueError("Unknown artifact kind")
    values = slider.state_dict() if state is None else state
    if kind == TEACHER and slider.ordinary:
        raise ValueError("An ordinary LoRA cannot be exported as a particle teacher")
    if kind == TEACHER or slider.ordinary:
        validate_state(values, slider.state_dict())
    else:
        expected = {k: v for k, v in slider.state_dict().items()
                    if k != "particles" and ".bridge." not in k}
        validate_state(values, expected)
    record = dict(format=kind, upstream_revision=REVISION, rank=8, alpha=8,
                  base=slider.spec, calibration=calibration, training=training)
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in values.items()}
    with store.writing(name, sum(t.numel()*t.element_size() for t in tensors.values()) + 65536) as path:
        save_file(tensors, str(path), metadata={"glimmer_slider": json.dumps(record, sort_keys=True)})


def load_adapter(model, identity, path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Portable adapter exceeds 512 MiB limit")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        record = json.loads((handle.metadata() or {}).get("glimmer_slider", "{}"))
    if (record.get("format") not in (TEACHER, STUDENT) or record.get("upstream_revision") != REVISION
            or record.get("rank") != 8 or record.get("alpha") != 8
            or record.get("base") != signature(model, identity)):
        raise ValueError("Artifact format, upstream or base identity mismatch")
    state = load_file(str(path), device="cpu")
    network = Slider(model, identity, ordinary=record["format"] == STUDENT)
    try:
        validate_state(state, network.state_dict())
        network.load_state_dict(state, strict=True)
    except Exception:
        network.detach()
        raise
    return network, record
