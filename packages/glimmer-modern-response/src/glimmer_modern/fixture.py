"""Small causal CPU model with Glimmer's grouped-KV attention target geometry.

This is a protocol fixture, never a pretrained Glimmer model or quality result.
"""
from types import SimpleNamespace
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(12, 12, bias=False)
        self.k_proj = nn.Linear(12, 4, bias=False)
        self.v_proj = nn.Linear(12, 4, bias=False)
        self.o_proj = nn.Linear(12, 12, bias=False)

    def forward(self, value):
        mixed = self.q_proj(value) + self.k_proj(value).repeat(1, 1, 3) + self.v_proj(value).repeat(1, 1, 3)
        causal = mixed.cumsum(1) / torch.arange(1, value.shape[1] + 1).view(1, -1, 1)
        return value + self.o_proj(causal.tanh())


class TinyGlimmerCPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(32, 12)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = Attention()
        self.model.language_model.layers = nn.ModuleList([layer])
        self.model.vision_model = nn.Linear(12, 12)

    def forward(self, input_ids, **kwargs):
        value = self.embedding(input_ids)
        for layer in self.model.language_model.layers:
            value = layer.self_attn(value)
        return SimpleNamespace(hidden_states=(value,))


def make_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyGlimmerCPU().eval()
