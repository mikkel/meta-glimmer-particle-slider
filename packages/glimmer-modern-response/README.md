# Opt-in modern Glimmer response cohort

This separate distribution uses `glimmer_modern` and the separate registry name
`glimmer-modern`. Install it in a fresh environment, never in the historical
PG82 environment. The repository owner remains `mikkel`.

The complete factory delegates updates, optimizers, penalties, sampling, EMA,
checkpoint validation and native-only export to the shared particle-sliders
response lifecycle. HyperGAN's verified builder executes both HNDL routed
projections and the HNDL readout critic. One real 128x4 bank participates in all
text attention branches. Q/O and grouped K/V retain their different dimensions;
the response is the genuine final text hidden state, without projection,
pooling, truncation, token reinterpretation or spatial reshaping.

The explicit new objective cohort is the shared released PG0.9.4 conditional
BISPF recipe. It does **not** claim numerical objective equivalence to PG82's
historical winning formulation. Network graph parity is tested separately.
The historical loader, trainer, RAW1024 artifacts and distilled PEFT students
are unchanged and require their original source pins and loaders.

The current cohort is `glimmer-modern-final-hidden-hg-hndl-v2-cpu`, with explicit
initialization policy `cpu-sha256-component-streams-v1`. The declared run `seed`
initializes the bank, every named adapter projection, and the critic using
separate SHA256-derived CPU streams before their random constructors. The
caller CPU RNG is restored on success and failure. CUDA RNG is neither queried
nor seeded by this CPU-only policy, so existing CUDA state is untouched and no
GPU initialization is requested. `model_seed` separately retains the original
frozen base mapping. The policy and run seed are bound in configuration,
backend metadata, graph identities and native-only inference specifications.

Earlier v1 packets require their original source revision and initialization
behavior. They are refused by v2; no checkpoint relabeling or migration occurs.
The shared loss, bank architecture and frozen positive/neutral hidden-response
semantics are unchanged.

## Current admission

Only `mode="cpu-fixture"`, width12, FP32 CPU is admitted. The bundled causal
model is a synthetic protocol fixture, not pretrained Glimmer. Native model
loading/training, width>=170, media/token generation and model quality are
unqualified and refused before preparation. No native admission option or
continuous-domain override exists. Passing CPU tests cannot open these gates.

The provisional shared-core source is `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`
(draft PR148), official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0.
`source-lock.json` verifies installed shared sources; the common builder checks
the complete HNDL closure, HyperGAN closure and clean build provenance before
any native attachment. Final central core admission remains a review blocker;
updated lifecycle corrections require a new audited lock and CPU receipt.

Borrowed CPU models with existing local module/parameter hooks, instance call
or serialization overrides, compiled calls or active checkpoint wrappers are
refused before model hashing or attachment. Cleanup restores original instance
attribute presence, including inherited forwards and checkpoint attributes.
These product checks do not provide packet-aware admission: deep packet/source
and tensor checks may still occur after factory preparation in the provisional
shared lifecycle. Universal pre-hook refusal is not claimed.

## CPU use

Create a fresh Python3.11–3.14 environment and install CPU PyTorch from the
public CPU index, then install this directory. It does not depend on the
historical `meta-glimmer-particle-slider` distribution.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install './packages/glimmer-modern-response[test]'
python -m pytest packages/glimmer-modern-response/tests -q
```

Store a bounded local JSON file with `train` and `evaluation` arrays, each with
2..32 pairs such as `{"neutral":[1,3],"positive":[2,4],"history":[6,7]}`.
IDs belong to the fixture's 32-token vocabulary. Each pair shares exactly the
same continuation history. Every split/target and the frozen base weight bytes
are bound to the continuation identity. Use fixed distinct held-out rows.

```python
from glimmer_modern.lifecycle import cpu_config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

config = cpu_config('pairs.json')
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint('modern-training.pt')
with open_response(config, resume='modern-training.pt') as run:
    run.train()
    run.evaluate(split='evaluation', strength=.5)
    run.export('modern-raw.pt')
    run.export_inference('modern-ema-v2.pt', selection='ema')
with open_inference('modern-ema-v2.pt', operation='response',
                    native_request={'ids': [1, 3, 6, 7]}) as run:
    response = run.response(ids=[1, 3, 6, 7], strength=.5)
```

The last factory reconstructs only the deterministic CPU base and adapter. It
requires no paired data, teacher cache, critic, optimizer or training factory.
Training and inference use the shared versioned schemas and reject historical
packets, wrong cohorts, sources and tensor contracts. A response is a numerical
hidden-state probe, not a native text quality or rendering result.
