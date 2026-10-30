"""Optional inference-only programs over the original raw routed modules.

No fitted weights, reduced precision, changed routing, merged projections, or
algebraic reassociation. Freeze removes Python module dispatch and constant
transpose work. Numerical graph optimizations are explicitly disabled. Native
CUDA equivalence and timing must still pass before this option is promoted.
"""
import torch
from torch import nn


class RawParticleProgram(nn.Module):
    def __init__(self, branch, particles):
        super().__init__()
        self.down, self.up, self.bridge = branch.lora_down, branch.lora_up, branch.bridge
        self.register_buffer('particles', particles.detach())

    def forward(self, x):
        return self.up(self.bridge(self.down(x.to(self.down.weight)), self.particles))


def freeze_raw_particle(branch, particles):
    if branch.training or particles.requires_grad or any(p.requires_grad for p in branch.parameters()):
        raise ValueError('Raw particle programs require frozen inference parameters')
    # Do not freeze the host or change the resident adapter's state_dict.
    program = RawParticleProgram(branch, particles).eval()
    return torch.jit.freeze(torch.jit.script(program), optimize_numerics=False)
