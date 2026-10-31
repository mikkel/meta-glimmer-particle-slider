"""Validate installed source bytes before constructing or attaching native objects."""
import hashlib
import json
from pathlib import Path


def verify_sources():
    import particle_sliders
    from particle_sliders.fibo_hndl import verified_builder
    root = Path(particle_sliders.__file__).resolve().parent
    lock = json.loads(Path(__file__).with_name("source-lock.json").read_text())
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".hndl")}
    if actual != lock["core_sources"]:
        raise ValueError("installed shared-core source differs; use the original cohort")
    builder, identity = verified_builder()
    return builder, dict(identity, core_commit=lock["core_commit"])
