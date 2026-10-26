"""Text-tower attachment of upstream rank-8 routed particle branches."""
from contextlib import contextmanager
import math
import re

import torch
from torch import nn

from .upstream import api

TARGET = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")


def projections(model):
    selected = {name: module for name, module in model.named_modules() if TARGET.fullmatch(name)}
    if not selected:
        raise ValueError("No Glimmer language attention projections found")
    layers = {}
    for name, module in selected.items():
        if type(module) is not nn.Linear or module.weight.device.type != "cpu":
            raise ValueError("This release qualifies unquantized CPU Linear hosts only")
        if hasattr(module, "_glimmer_slider"):
            raise ValueError("An adapter is already attached")
        index, part = TARGET.fullmatch(name).groups()
        layers.setdefault(int(index), set()).add(part)
    if sorted(layers) != list(range(len(layers))) or any(len(parts) != 4 for parts in layers.values()):
        raise ValueError("Require complete contiguous text attention layers")
    return selected


def signature(model, identity):
    if set(identity) != {"model_id", "revision"} or not all(isinstance(x, str) and x for x in identity.values()):
        raise ValueError("Explicit base model ID and immutable revision required")
    if not re.fullmatch(r"[0-9a-f]{40}", identity["revision"]):
        raise ValueError("Base revision must be a full commit hash")
    return dict(identity=identity, projections={n: [m.in_features, m.out_features]
                                               for n, m in projections(model).items()})


def branch_type():
    # Inherit the project's ordinary LoRA A/B initialization and application.
    class ParticleBranch(api().LoRAModule):
        def __init__(self, name, module, particles):
            super().__init__(name, module, multiplier=0., lora_dim=8, alpha=8)
            # Anima's particle branch initialization; ordinary export retains U.
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            object.__setattr__(self, "cloud", particles)
            self.bridge = api().stamp.bridge()

        def routed(self, x):
            weight = self.lora_down.weight
            return self.bridge(self.lora_down(x.to(weight)), self.cloud)

        def forward(self, x):
            base = self.org_forward(x)
            if self.multiplier == 0:
                return base
            return base + self.lora_up(self.routed(x)).to(base) * (self.multiplier * self.scale)
    return ParticleBranch


def plain_type():
    class PlainBranch(api().LoRAModule):
        def forward(self, x):
            if self.multiplier == 0:
                return self.org_forward(x)
            return super().forward(x)
    return PlainBranch


class Slider(nn.Module):
    """One shared learned bank, all-particle routing, no random particle at sample time.

    Host ownership remains with caller. The registered parameters are adapter-only.
    CPU-only is deliberate until this new implementation has native qualification.
    """
    def __init__(self, model, identity, *, ordinary=False):
        super().__init__()
        self.spec = signature(model, identity)
        targets = projections(model)
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("All host parameters must be on CPU")
        self.ordinary = bool(ordinary)
        self.branches = nn.ModuleDict()
        self._targets = targets
        self._base_flags = [(p, p.requires_grad) for p in model.parameters()]
        model.requires_grad_(False)
        if not ordinary:
            self.particles = nn.Parameter(torch.randn(128, 4))
        cls = plain_type() if ordinary else branch_type()
        for index, (name, module) in enumerate(targets.items()):
            key = str(index)
            branch = (cls(name, module, multiplier=0., lora_dim=8, alpha=8) if ordinary
                      else cls(name, module, self.particles))
            branch.apply_to()
            # org_module has been removed, so this cannot cast the base weights.
            branch.to(device="cpu", dtype=torch.float32)
            self.branches[key] = branch
            object.__setattr__(module, "_glimmer_slider", self)
        self.strength = 0.

    @contextmanager
    def scaled(self, strength):
        value = float(strength)
        if not math.isfinite(value):
            raise ValueError("Strength must be finite")
        previous = self.strength
        self.strength = value
        for branch in self.branches.values():
            branch.multiplier = value
        try:
            yield
        finally:
            self.strength = previous
            for branch in self.branches.values():
                branch.multiplier = previous

    def detach(self):
        for module, branch in zip(self._targets.values(), self.branches.values()):
            module.forward = branch.org_forward
            del module._glimmer_slider
        for parameter, enabled in self._base_flags:
            parameter.requires_grad_(enabled)

    @contextmanager
    def using_state(self, state):
        saved = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict(state, strict=True)
        try:
            yield
        finally:
            self.load_state_dict(saved, strict=True)


@torch.no_grad()
def distill(slider, model, calibration_inputs, *, max_rows=128):
    """Capture bounded actual host inputs, call unchanged upstream dual ridge fit.

    The retained teacher U and fitted A form an approximate ordinary rank-8 LoRA.
    The caller detaches the teacher before attaching/loading the resulting student.
    """
    if slider.ordinary or not 2 <= max_rows <= 512:
        raise ValueError("Require a particle teacher and 2..512 calibration rows")
    captured = {key: [] for key in slider.branches}
    hooks = []
    for key, module in zip(slider.branches, slider._targets.values()):
        def capture(_module, args, key=key):
            count = sum(len(x) for x in captured[key])
            if count < max_rows:
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                captured[key].append(x[:max_rows-count].float().cpu().clone())
        hooks.append(module.register_forward_pre_hook(capture))
    try:
        with slider.scaled(1.):
            for inputs in calibration_inputs:
                model(input_ids=inputs, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    state, metrics = {}, {}
    for key, branch in slider.branches.items():
        if not captured[key] or sum(len(x) for x in captured[key]) < 2:
            raise ValueError("Insufficient calibration activations")
        x = torch.cat(captured[key])
        routed = branch.routed(x)
        down, ridge = api().fit_routed_down(x, routed, ridge_fraction=.01)
        prefix = f"branches.{key}."
        state[prefix + "lora_down.weight"] = down.contiguous()
        state[prefix + "lora_up.weight"] = branch.lora_up.weight.detach().clone()
        state[prefix + "alpha"] = branch.alpha.detach().clone()
        metrics[key] = dict(rows=len(x), ridge=float(ridge),
                            bottleneck_rms=float((x @ down.T-routed).square().mean().sqrt()))
    return state, metrics
