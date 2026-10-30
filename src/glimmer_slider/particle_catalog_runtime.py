"""Atomic particle catalog replacement on one resident base, between requests."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from .sampler import _Adapter, AdapterSnapshot
from .storage import TEACHER
from .upstream import REVISION


@torch.no_grad()
def replace_particles(bank, artifacts, *, probe=None):
    if len({a.id for a in artifacts}) != len(artifacts) or not artifacts:
        raise ValueError('Particle catalog IDs must be distinct and nonempty')
    if bank._in_use.get(): raise RuntimeError('Catalog replacement inside generation')
    with bank._catalog:
        bank._catalog.wait_for(lambda: not bank._refreshing and bank._readers == 0)
        bank._refreshing = True
        previous = bank.snapshot()
    try:
        adapters = {}
        devices = [bank.device.index or 0] if bank.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            for artifact in artifacts:
                path = Path(artifact.path)
                if artifact.kind != TEACHER or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                    raise ValueError('Particle artifact hash or kind differs')
                with safe_open(str(path), framework='pt', device='cpu') as source:
                    record = json.loads((source.metadata() or {})['glimmer_slider'])
                training = record.get('training') or {}
                if (record.get('base') != bank.spec or record.get('upstream_revision') != REVISION
                        or record.get('format') != TEACHER or record.get('rank') != 8 or record.get('alpha') != 8
                        or training.get('weights') != 'raw' or training.get('step') != artifact.step):
                    raise ValueError('Particle base, routing schema or raw step differs')
                existing = next((a for a in previous.artifacts if a.id == artifact.id), None)
                if existing and existing.sha256 == artifact.sha256:
                    adapters[artifact.id] = previous.adapters[artifact.id]
                else:
                    adapters[artifact.id] = _Adapter(bank.targets, load_file(str(path), device='cpu'), particle=True, device=bank.device)
                if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                    raise ValueError('Particle artifact changed while staging')
            proposed = AdapterSnapshot(previous.revision + 1, tuple(artifacts), MappingProxyType(adapters))
            checks = {}
            if probe is not None:
                with bank._plan({}, previous): before = probe()
                with bank._plan({}, proposed): after = probe()
                if not torch.equal(before, after): raise ValueError('Catalog replacement changed OFF')
                for artifact in artifacts:
                    with bank._plan({artifact.id: 1.}, proposed): active = probe()
                    if not torch.isfinite(active).all(): raise ValueError('Nonfinite particle logits')
                checks = dict(zero_exact=True, active_finite=len(artifacts))
        with bank._catalog:
            bank.adapters, bank.artifacts, bank.revision = adapters, tuple(artifacts), proposed.revision
            with bank._cache_lock: bank._cache.clear()
        return dict(revision=bank.revision, ids=[a.id for a in artifacts], checks=checks)
    finally:
        with bank._catalog:
            bank._refreshing = False; bank._catalog.notify_all()
