"""Lazy positioned reads avoid mapping an entire large checkpoint shard."""
from contextlib import contextmanager, ExitStack
from pathlib import Path

from safetensors import safe_open


@contextmanager
def pread_state(root, filenames):
    root = Path(root)
    with ExitStack() as stack:
        state = {}
        for name in filenames:
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("Unsafe safetensors filename")
            handle = stack.enter_context(safe_open(root/name, framework="pt", device="cpu", backend="pread"))
            for key in handle.keys():
                if key in state:
                    raise ValueError("Duplicate tensor across checkpoint shards")
                state[key] = handle.get_slice(key)
        yield state
