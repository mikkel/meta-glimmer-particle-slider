"""Versioned CPU initialization streams; never seed or initialize CUDA."""
from contextlib import contextmanager
import hashlib

import torch

from . import INITIALIZATION_POLICY


@contextmanager
def initialization_rng(seed, component=None):
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("initialization seed must be an integer in [0, 2**63-100)")
    # The base keeps its original model_seed mapping. New adapter/bank/critic
    # streams are separated by stable names, independent of constructor order.
    value = seed if component is None else int.from_bytes(hashlib.sha256(
        f"{INITIALIZATION_POLICY}:{seed}:{component}".encode()).digest()[:8], "big") % 2**63
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(value).get_state())
        yield
