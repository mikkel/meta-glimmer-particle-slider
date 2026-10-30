"""Resident signed adapter stacks for the Glimmer text attention projections.

NF4 weights are never merged or rewritten. Ordinary branches are packed as
A=cat(A_i, dim=0), B=cat(s_i B_i, dim=1): B(Ax)=sum_i s_i B_i A_i x.
Particle branches retain their upstream nonlinear routing and are added
independently. A context-local immutable plan avoids shared multiplier state.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,replace
import hashlib
import json
import math
import re
from pathlib import Path
import threading
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .host import projections, signature, branch_type, plain_type
from .storage import STUDENT, TEACHER, validate_state
from .upstream import REVISION


@dataclass(frozen=True)
class Artifact:
    id: str
    title: str
    path: str
    sha256: str
    kind: str
    step: int
    qualification: str
    quality_note: str = ''
    ordinary_to_particle_rms: float | None = None
    source_teacher_sha256: str | None = None
    definition_version: int | None = None
    quality_evidence_sha256: str | None = None


@dataclass(frozen=True)
class AdapterSnapshot:
    revision: int
    artifacts: tuple
    adapters: object


def verify_final_ordinary(artifact):
    """Require hash-bound native export qualification, never a UI label alone."""
    path=Path(artifact.path)
    if artifact.kind!=STUDENT or artifact.step!=256 or path.name!='ordinary-lora-raw-000256.safetensors':
        raise ValueError('Registration requires the final raw ordinary step-256 export')
    if path.stat().st_size>512*1024**2:
        raise ValueError('Registered adapter exceeds the portable size cap')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact.sha256:
        raise ValueError('Qualified artifact hash changed')
    ledger=json.loads((path.parent/'ordinary-validation.json').read_text())
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        record=json.loads((handle.metadata() or {}).get('glimmer_slider','{}'))
    if not isinstance(ledger,dict) or not isinstance(record,dict):
        raise ValueError('Malformed ordinary export qualification record')
    identity=ledger.get('identity') or {}
    outputs=ledger.get('outputs',{})
    if not isinstance(identity,dict) or not isinstance(outputs,dict) or any(not isinstance(row,dict) for row in outputs.values()):
        raise ValueError('Malformed ordinary export qualification outputs')
    training=record.get('training') or {}
    if (record.get('format')!=STUDENT or training.get('step')!=256 or training.get('weights')!='raw'
            or ledger.get('ordinary_reload_exact') is not True or ledger.get('strength_zero_exact_all3') is not True
            or ledger.get('student_sha256')!=artifact.sha256 or len(outputs)!=3
            or identity.get('weights')!='raw' or identity.get('format')!=STUDENT
            or (record.get('calibration') or {}).get('identity')!=identity
            or any(row.get('artifact_sha256')!=artifact.sha256 or row.get('weights')!='raw'
                   or not row.get('token_ids') for row in outputs.values())):
        raise ValueError('Final ordinary export qualification is incomplete or mismatched')
    return ledger


def describe(path, *, id, title, qualification):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 512*1024**2:
        raise ValueError('Adapter exceeds portable size cap')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        record = json.loads((handle.metadata() or {}).get('glimmer_slider', '{}'))
    if record.get('format') not in (STUDENT, TEACHER):
        raise ValueError('Unknown adapter format')
    training = record.get('training') or {}
    if training.get('weights') != 'raw' or not isinstance(training.get('step'), int):
        raise ValueError('Only explicitly raw, step-labeled artifacts are admitted')
    return Artifact(id, title, str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                    record['format'], training['step'], qualification)


def discover(qualification_root):
    """Read-only snapshot. Prefer final raw ordinary LoRA; retain partial labels.

    No training continuations are unpickled. Only portable safetensors files
    published atomically by the trainer are inspected.
    """
    root = Path(qualification_root).resolve(strict=True)
    collection = root/'style-showcase-v1'
    gallery = json.loads((collection/'GALLERY-DATA.json').read_text())
    manifest_path=collection/'STUDIO-ARTIFACT-MANIFEST.json'
    manifest_bytes=manifest_path.read_bytes() if manifest_path.exists() else None
    manifest=json.loads(manifest_bytes) if manifest_bytes else {}
    by_sha={row['sha256']:row for row in manifest.get('entries',[]) if 'sha256' in row}
    found = []
    for style in gallery['styles']:
        sid = style['id']
        runs = sorted(collection.glob(('medieval' if sid == 'medieval-melodrama' else sid)+'-run-v*'),
                      key=lambda path:int(re.search(r'-v(\d+)$',path.name).group(1)) if re.search(r'-v(\d+)$',path.name) else -1)
        candidates = [p for run in reversed(runs) for p in sorted(run.glob('ordinary-lora-raw-*.safetensors'), reverse=True)]
        if not candidates and sid=='extreme-excitement':
            legacy=root/'j-distillation-v1/ordinary-lora-j256-raw.safetensors'
            if legacy.is_file():candidates=[legacy]
        if not candidates:
            candidates = [p for run in reversed(runs) for p in sorted(run.glob('teacher-raw-*.safetensors'), reverse=True)
                          if not p.name.endswith('-000000.safetensors')]
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            continue
        path = candidates[0]
        ledger_path = path.parent/'ordinary-validation.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        qualified = (ledger.get('ordinary_reload_exact') is True
                     and ledger.get('strength_zero_exact_all3') is True
                     and len(ledger.get('outputs', {})) == 3
                     and ledger.get('student_sha256') == hashlib.sha256(path.read_bytes()).hexdigest())
        label = ('Ordinary LoRA approximation: file reload and OFF checks passed'
                 if qualified else 'UNQUALIFIED ordinary LoRA: export validation record unavailable'
                 if 'ordinary-lora' in path.name else 'UNQUALIFIED particle checkpoint; exploratory only')
        artifact=describe(path, id=sid, title=style['title'], qualification=label)
        row=by_sha.get(artifact.sha256,{})
        if row.get('style_id')==sid and row.get('format')==artifact.kind and row.get('step')==artifact.step:
            checks=row.get('native_checks') or {}
            native_verified=(row.get('status')=='ordinary_raw_native_validated_at_alpha1'
                and checks.get('exact_zero_logits_cases',0)>=3
                and checks.get('ordinary_serialized_reload_exact_logits_cases',0)>=3
                and checks.get('ordinary_generation_cases',0)>=3)
            if native_verified:
                artifact=replace(artifact,qualification='Ordinary LoRA approximation: native export checks recorded at strength 1')
            rms=row.get('ordinary_to_particle_RMS')
            if not isinstance(rms,(int,float)) or not math.isfinite(rms) or rms<0:rms=None
            artifact=replace(artifact,quality_note='Outputs may differ substantially from the particle teacher. '+row.get('style_note',''),
                ordinary_to_particle_rms=rms,source_teacher_sha256=(row.get('source_particle_teacher') or {}).get('sha256'),
                definition_version=row.get('definition_version'),quality_evidence_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        found.append(artifact)
    return tuple(found)


class _Adapter(nn.Module):
    def __init__(self, targets, state, *, particle, device):
        super().__init__()
        self.particle = particle
        if particle:
            self.particles = nn.Parameter(torch.empty(128, 4))
        self.branches = nn.ModuleDict()
        cls = branch_type() if particle else plain_type()
        for i, (name, module) in enumerate(targets.items()):
            part = (cls(name, module, self.particles) if particle
                    else cls(name, module, multiplier=0., lora_dim=8, alpha=8))
            # Never apply_to(): the bank owns the single wrapper on the base.
            del part.org_module
            self.branches[str(i)] = part
        validate_state(state, self.state_dict())
        self.load_state_dict(state, strict=True)
        self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


class ResidentSliders:
    def __init__(self, model, identity, artifacts, *, execution='cpu', cache_size=2,
                 particle_execution='eager'):
        if particle_execution not in ('eager', 'scripted'):
            raise ValueError('Unknown raw particle execution mode')
        self.particle_execution = particle_execution
        if len({a.id for a in artifacts}) != len(artifacts):
            raise ValueError('Duplicate slider ID')
        if not 1 <= cache_size <= 4:
            raise ValueError('Combination cache must be bounded to 1..4 entries')
        self.artifacts = tuple(artifacts)
        self.targets = projections(model, execution=execution)
        self.spec = signature(model, identity, execution=execution)
        self.adapters = {}
        self._active = ContextVar('glimmer_combination', default=None)
        self._in_use = ContextVar('glimmer_generation_active',default=False)
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()
        self._catalog = threading.Condition(threading.RLock())
        self._readers = 0
        self._refreshing = False
        self.revision = 0
        self._originals = {}
        self.device = next(model.parameters()).device
        # Complete validation happens before the first base forward is changed.
        for artifact in self.artifacts:
            path = Path(artifact.path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                raise ValueError('Artifact changed since catalog snapshot')
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                record = json.loads((handle.metadata() or {}).get('glimmer_slider', '{}'))
            if (record.get('base') != self.spec or record.get('upstream_revision') != REVISION
                    or record.get('format') != artifact.kind or record.get('rank') != 8
                    or record.get('alpha') != 8):
                raise ValueError('Adapter base, upstream, rank or format mismatch')
            state = load_file(str(path), device='cpu')
            self.adapters[artifact.id] = _Adapter(self.targets, state, particle=artifact.kind == TEACHER, device=self.device)
        for i, (name, module) in enumerate(self.targets.items()):
            original = module.forward
            self._originals[name] = original
            def forward(x, original=original, index=i):
                base = original(x)
                plan = self._active.get()
                if not plan:
                    return base
                packed, particles = plan[index]
                cast = x.to(dtype=torch.float32)
                delta = None
                if packed is not None:
                    down, up = packed
                    delta = F.linear(F.linear(cast, down), up)
                for evaluate, strength in particles:
                    if self.particle_execution == 'scripted':
                        addition = evaluate(cast) * strength
                    else:
                        addition = evaluate.lora_up(evaluate.routed(cast)) * strength
                    delta = addition if delta is None else delta + addition
                return base if delta is None else base + delta.to(base)
            module.forward = forward
            object.__setattr__(module, '_glimmer_slider', self)

    def snapshot(self):
        with self._catalog:
            return AdapterSnapshot(self.revision,self.artifacts,MappingProxyType(dict(self.adapters)))

    def strengths(self, values, snapshot=None):
        snapshot=snapshot or self.snapshot()
        if not isinstance(values, dict) or set(values)-set(snapshot.adapters):
            raise ValueError('Unknown slider ID')
        result = []
        for artifact in snapshot.artifacts:  # Catalog order, independent of JSON order.
            value = values.get(artifact.id, 0.)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('Strength must be numeric')
            strength = float(value)
            if not math.isfinite(strength) or abs(strength) > 8:
                raise ValueError('Signed strengths must be finite and within -8..8')
            if strength != 0:
                result.append((artifact.id, strength))
        return tuple(result)

    @torch.no_grad()
    def compile(self, values, snapshot=None):
        snapshot=snapshot or self.snapshot()
        strengths = self.strengths(values,snapshot)
        if not strengths:
            return None  # Exact original forward, including NF4 rounding.
        hashes={a.id:a.sha256 for a in snapshot.artifacts}
        key=tuple((sid,strength,hashes[sid]) for sid,strength in strengths)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            plan = []
            for i in range(len(self.targets)):
                down, up, particles = [], [], []
                for sid, strength in strengths:
                    adapter = snapshot.adapters[sid]
                    part = adapter.branches[str(i)]
                    if adapter.particle:
                        if self.particle_execution == 'scripted':
                            from .particle_inference import freeze_raw_particle
                            # Adapter identity owns this cache, so old snapshots
                            # retain their own program across catalog replacement.
                            if not hasattr(adapter, '_raw_programs'):
                                adapter._raw_programs = {}
                            if i not in adapter._raw_programs:
                                adapter._raw_programs[i] = freeze_raw_particle(part, adapter.particles)
                            evaluate = adapter._raw_programs[i]
                        else:
                            evaluate = part
                        particles.append((evaluate, strength * part.scale))
                    else:
                        down.append(part.lora_down.weight)
                        up.append(part.lora_up.weight * (strength * part.scale))
                packed = (torch.cat(down, 0), torch.cat(up, 1)) if down else None
                plan.append((packed, tuple(particles)))
            plan = tuple(plan)
            self._cache[key] = plan
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return plan

    @contextmanager
    def _plan(self, values, snapshot):
        token = self._active.set(self.compile(values,snapshot))
        try:
            yield
        finally:
            self._active.reset(token)

    @contextmanager
    def using(self, values, snapshot=None):
        with self._catalog:
            self._catalog.wait_for(lambda:not self._refreshing)
            self._readers+=1
            snapshot=snapshot or self.snapshot()
        try:
            in_use=self._in_use.set(True)
            with self._plan(values,snapshot):
                yield
        finally:
            self._in_use.reset(in_use)
            with self._catalog:
                self._readers-=1
                self._catalog.notify_all()

    @torch.no_grad()
    def register(self, artifact, *, probe=None, reload_adapter=False):
        """Stage and publish one qualified adapter; no base detachment or load.

        Server calls this only as a serial queue control job. Readers retain
        immutable snapshots; callers waiting behind a refresh keep old versions.
        A failed stage/probe never publishes a partially loaded adapter.
        """
        if self._in_use.get():
            raise RuntimeError('Adapter registration cannot run inside generation')
        if artifact.id not in self.adapters:
            raise ValueError('Registration may replace only an existing slider ID')
        verify_final_ordinary(artifact)
        with self._catalog:
            self._catalog.wait_for(lambda:not self._refreshing and self._readers==0)
            self._refreshing=True
            previous=self.snapshot()
        try:
            existing=next(a for a in previous.artifacts if a.id==artifact.id)
            if existing.sha256==artifact.sha256 and not reload_adapter:
                with self._catalog:
                    if existing!=artifact:
                        self.artifacts=tuple(artifact if a.id==artifact.id else a for a in previous.artifacts)
                        self.revision+=1
                return dict(changed=False,reloaded=False,revision=self.revision,sha256=artifact.sha256)
            path=Path(artifact.path)
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                record=json.loads((handle.metadata() or {}).get('glimmer_slider','{}'))
            if (record.get('base')!=self.spec or record.get('upstream_revision')!=REVISION
                    or record.get('format')!=STUDENT or record.get('rank')!=8 or record.get('alpha')!=8):
                raise ValueError('Registered adapter base, upstream or rank differs')
            devices=[self.device.index or 0] if self.device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                adapter=_Adapter(self.targets,load_file(str(path),device='cpu'),particle=False,device=self.device)
                verify_final_ordinary(artifact)  # Reject a file/ledger changed during staging.
                adapters=dict(previous.adapters);adapters[artifact.id]=adapter
                artifacts=tuple(artifact if a.id==artifact.id else a for a in previous.artifacts)
                proposed=AdapterSnapshot(previous.revision+1,artifacts,MappingProxyType(adapters))
                checks={}
                if probe is not None:
                    with self._plan({},previous): baseline=probe()
                    with self._plan({artifact.id:.25},proposed): active=probe()
                    with self._plan({},proposed): off=probe()
                    if not torch.equal(baseline,off) or not torch.isfinite(active).all():
                        raise ValueError('Staged adapter failed exact OFF or finite active logits')
                    checks=dict(zero_exact=True,active_finite=True,active_changed=not torch.equal(active,baseline))
            with self._catalog:
                self.adapters=adapters;self.artifacts=artifacts;self.revision=proposed.revision
                with self._cache_lock: self._cache.clear()
            return dict(changed=existing.sha256!=artifact.sha256,reloaded=True,revision=proposed.revision,
                sha256=artifact.sha256,checks=checks)
        except BaseException:
            with self._cache_lock: self._cache.clear()
            raise
        finally:
            with self._catalog:
                self._refreshing=False
                self._catalog.notify_all()

    def detach(self):
        for name, module in self.targets.items():
            module.forward = self._originals[name]
            del module._glimmer_slider
        self._cache.clear()
