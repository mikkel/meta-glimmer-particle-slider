"""Glimmer target/response bindings; all routed network operations stay shared."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from particle_sliders.fibo_hndl import HNDLParticleProjection
from particle_sliders.lifecycle import native_cleanup
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter

from . import PROTOCOL
from .provenance import verify_sources

TARGET = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")


def targets(model):
    from .fixture import TinyGlimmerCPU
    if type(model) is not TinyGlimmerCPU or model.config.hidden_size != 12:
        raise ValueError("native/wide Glimmer adapters await central qualification; CPU fixture only")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("CPU fixture base must be entirely FP32 CPU")
    selected = {name: module for name, module in model.named_modules() if TARGET.fullmatch(name)}
    layers = {}
    for name, module in selected.items():
        if type(module) is not nn.Linear or module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            raise ValueError("modern CPU cohort requires genuine FP32 CPU linear projections")
        if hasattr(module, "_glimmer_slider") or hasattr(module, "_glimmer_modern"):
            raise ValueError("projection already has an attached slider")
        index, part = TARGET.fullmatch(name).groups()
        layers.setdefault(int(index), set()).add(part)
    if not layers or sorted(layers) != list(range(len(layers))) or any(len(parts) != 4 for parts in layers.values()):
        raise ValueError("complete contiguous Glimmer text attention projections required")
    return selected


def model_identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return dict(model_id="glimmer-modern/cpu-protocol-fixture", weights_sha256=digest.hexdigest(),
                qualification="synthetic CPU only; no native model weights")


class Slider(nn.Module):
    def __init__(self, model):
        super().__init__()
        selected = targets(model)
        builder, identity = verify_sources()  # Before any model mutation.
        self.identity = identity
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.branches = nn.ModuleDict()
        # Build every graph before changing any native forward or gradient flag.
        for index, (name, module) in enumerate(selected.items()):
            self.branches[str(index)] = HNDLParticleProjection(name, module, self.particles,
                8, 8, builder=builder, identity=identity, attach=False)
        self._targets = selected
        self._restore = native_cleanup(model)
        self._modes = [(m, m.training) for m in model.modules()]
        self._flags = [(p, p.requires_grad) for p in model.parameters()]
        self.strength = 0.
        self._attached = []
        self.closed = False
        try:
            model.eval().requires_grad_(False)
            for module, branch in zip(selected.values(), self.branches.values()):
                self._attached.append((module, branch))
                module.forward = branch.forward
                object.__setattr__(module, "_glimmer_modern", self)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("modern unipolar strength must be in [0,1]")
        previous = self.strength
        self.strength = float(strength)
        for branch in self.branches.values():
            branch.multiplier = float(strength)
        try:
            yield
        finally:
            self.strength = previous
            for branch in self.branches.values():
                branch.multiplier = previous

    def close(self):
        if self.closed:
            return
        first_error = None
        failed = []
        for module, branch in reversed(self._attached):
            try:
                if module.forward != branch.forward or getattr(module, "_glimmer_modern", None) is not self:
                    raise RuntimeError("native projection ownership changed during modern session")
                module.forward = branch.org_forward
                del module._glimmer_modern
            except BaseException as error:
                failed.append((module, branch))
                if first_error is None:
                    first_error = error
        self._attached = list(reversed(failed))
        try:
            self._restore()
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            # A checkpoint cleanup failure must not skip modes or gradients.
            for module, mode in self._modes:
                module.training = mode
            for parameter, enabled in self._flags:
                parameter.requires_grad_(enabled)
        self.closed = first_error is None
        if first_error is not None:
            raise first_error


class GlimmerResponse(ARLastHiddenResponseAdapter):
    family = "glimmer_text_final_hidden"

    def _hidden(self, ids):
        ids = torch.tensor([ids], dtype=torch.long)
        output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            output_hidden_states=True, use_cache=False)
        if not getattr(output, "hidden_states", None):
            raise ValueError("genuine differentiable final text hidden states required")
        return output.hidden_states[-1][:, -1]


class Native:
    def __init__(self, model):
        targets(model)  # Reject unqualified geometry before hashing or mutation.
        self.model = model
        self.identity = model_identity(model)
        self.slider = Slider(model)
        try:
            self.backend = GlimmerResponse(model, self.slider, model_identity=self.identity)
            self.network = self.backend.network
        except BaseException:
            self.slider.close()
            raise

    def generated(self, prepared):
        return self.backend.generated(prepared)

    def metadata(self):
        return dict(protocol=PROTOCOL, model_identity=self.identity,
                    response_domain="continuous_vector", response_shape=[1, self.model.config.hidden_size],
                    source_profile=self.slider.identity,
                    target="frozen positive final hidden; identical causal continuation history")

    def response(self, *, ids, strength=1., seed=0):
        validate_ids(ids)
        return self.backend._response(tuple(ids), scale=strength).detach().clone()

    @contextmanager
    def observation(self):
        # The common lifecycle owns adapter values/modes and RNG. Native model
        # state and gradients belong to this product, including borrowed models.
        restore = native_cleanup(self.model)
        buffers = [(b, b.detach().clone()) for b in self.model.buffers()]
        parameters = list(self.model.parameters()) + list(self.network.parameters())
        gradients = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone())
                     for p in parameters]
        try:
            self.model.eval()
            yield
        finally:
            try:
                with torch.no_grad():
                    for buffer, value in buffers:
                        buffer.copy_(value)
                    for parameter, enabled, grad, value in gradients:
                        parameter.requires_grad_(enabled)
                        parameter.grad = grad
                        if grad is not None:
                            grad.copy_(value)
            finally:
                restore()

    def close(self):
        try:
            self.backend.close_checkpointing()
        finally:
            self.slider.close()


def validate_ids(ids):
    if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 32
            or any(type(i) is not int or not 0 <= i < 32 for i in ids)):
        raise ValueError("fixture requires 1..32 token IDs within its 32-token vocabulary")


class ReadoutCritic(nn.Module):
    """Exact shared AudioResponseCritic topology executed through HG/HNDL."""
    score_bound = 0

    def __init__(self, width, condition_width):
        super().__init__()
        builder, identity = verify_sources()
        source = Path(__file__).with_name("readout.hndl").read_text()
        self.graph = builder(source, input_shape={"response": ["B", width],
            "condition": ["B", condition_width]}, output_shape=["B", 1])
        self.identity = dict(identity, readout_sha256=hashlib.sha256(source.encode()).hexdigest())

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)
