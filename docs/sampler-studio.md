# Private particle Slider Studio

The studio keeps one locally provisioned Glimmer base resident and applies raw
particle adapters without merging NF4 weights. It serves only on loopback or an
operator-configured Tailscale address. It is not a hosted service, authentication
system, model downloader, training job or ordinary-LoRA loader.

## Setup

Install the package with its `native` dependencies for GPU serving. The upstream
particle-sliders revision and algorithm hashes are checked by the existing loader.
Stage and independently verify the official local Glimmer snapshot first, as in
the main README. No weight files, training checkpoints, credentials or private
catalogs are distributed here.

Copy `examples/studio.json` and `examples/studio-catalog.example.json` to local
configuration files. Edit **every placeholder**. Catalog paths are relative to the
catalog file; configuration paths are relative to the configuration file. Each
entry must identify an existing `glimmer-particle-teacher-v1` safetensors artifact,
its actual SHA256, raw training step, title and honest qualification note. All
artifacts must match the loaded base identity, projection shapes, upstream
revision, rank and alpha. Distilled ordinary LoRAs remain separate and are never
silently substituted.

The three coordination paths must be the same files used by other jobs sharing
the host/device. Different names do not coordinate. Native admission refuses
unowned allocations on the selected physical GPU. If a legitimate process must
coexist on that GPU, explicitly list its PID or systemd service in your **private**
`protected_processes` configuration, e.g. `[{"service":"your-worker.service"}]`.
The guardian freezes its PID, start time, executable and hash once; it stops if
that identity changes and never follows a replacement process automatically.
Do not share a private configuration or runtime receipts publicly.

```bash
python -m glimmer_slider.studio_guard --config path/to/local-studio.json
```

Use a fresh `run_dir` per invocation. The default example binds
`http://127.0.0.1:8789/` for two hours, with a finite maximum of eight hours. The
public launch wrapper is Linux-only (`/proc`, flock, systemd for protected service
entries, `nvidia-smi`, parent-death protection). It creates only its own process
and coordination/output files; it never changes swap, cgroup limits or services.
Stopping the guardian terminates its owned worker before releasing the leases.
It does not restart automatically. This portable wrapper has CPU fixtures and a bounded full-model inference smoke
recorded in [the native receipt](studio-native-smoke.json). It has not been
qualified for unattended production deployment. A busy listener is rejected
before the base model loads; readiness is written only after listener activation.

A model-free **real UI preview** can inspect verified adapter metadata and open
previously saved runs. It cannot generate and is labeled accordingly:

```bash
CUDA_VISIBLE_DEVICES='' python -m glimmer_slider.studio \
  --config path/to/local-studio.json --preview
```

The preview obeys the finite configured duration. It loads no base model and
requires explicit hidden CUDA devices. `Open run` reads a saved JSON file in the
browser only; recorded output is labeled as a saved run and is not a new inference
result. `Save run` retains exact settings, checkpoint provenance and output.

## Streaming and per-slider scopes

Generation is serialized. The browser receives actual incremental decoded token
callbacks as NDJSON, with separate reasoning, answer, raw and framing fields.
Reasoning+answer and direct-answer modes remain explicit. Reasoning and answer
share the output budget when reasoning is on. Limits are 4096 new tokens, 4096
prompt tokens, 8192 combined tokens, 240 seconds per generation and signed
strengths within −8..8; requests exceeding context are rejected, not clamped.
Cancellation and last-client disconnect cooperate at the next token boundary.
Repeated request IDs cannot accidentally create a second identical job.

Each slider independently selects **Thinking / Response / Both**, with **Both**
the backward-compatible default. Strength and scope are separate. Gating changes
the adapter execution plan, not merely visible text. Explicit `to=self` and
`to=user` markers identify the next prediction's channel, including recognizable
missing-separator output. Unknown/partial/closed channels keep Both active but
turn Thinking/Response-only sliders off. Boundary tokens use the preceding plan;
prompt prefill uses one plan for the input. Cached hidden states can carry earlier
influence, so Thinking can affect answers indirectly. This is not answer isolation
or a promise of channel intent for every malformed sequence.

The strict tokenizer parse is retained. Missing protocol separators are flagged;
only explicitly identifiable channels are recovered for display. Unclassified
text remains visible in the raw pane and is never invented as reasoning. A time
or token cap with no final text is shown as such. Response labels freeze the
server-confirmed slider names, exact strengths, scope, checkpoint steps and hashes;
changing controls or the catalog cannot relabel old output.

## Runtime/cache and safeguards

The library still defaults to eager particle execution. The example studio
explicitly selects the validated optional `scripted` program. It freezes the same
raw down/router/bridge/up operations with numerical optimizations disabled;
strength remains outside the program. All particles are routed on every input.
Programs belong to immutable adapter objects, are reused across strengths and
toggles, and are replaced for new checkpoint hashes. Mixture plans are bounded and
include checkpoint hashes. Catalog publication is atomic between requests; old
snapshots keep their original adapters. No base reload is needed for alpha changes.
TorchScript is deprecated in the installed PyTorch runtime. Use the explicit
`eager` configuration if incompatible; there is no silent fallback.

Prior operator measurements on the pinned full Glimmer host found median 1.155×
one-slider and 1.256× three-slider speedups across twelve paired comparisons.
Those comparisons passed exact prefill/cached-decode logits, tokens, text and
provenance checks. They used two prompts and two repeats; long-context behavior
and other hardware were not established. First use has compilation cost. These
are prior evidence for this core, not a fresh benchmark of this PR's launch wrapper.

Available RAM must be at least **23 GiB at admission** and **16 GiB at runtime**.
The startup value adds seven GiB loading headroom, rounded above a measured6.663GiB
owned-process loading peak, to the sixteen GiB reserve. These are conservative
host safeguards, not a universal statement of the model's memory requirement.
Missing RAM telemetry or any host OOM-counter change fails closed. Owned RSS,
faults, swap and I/O delay are attributed to the guardian and descendants but
remain diagnostic. Host PSI and the guardian's shared-app cgroup PSI are also
diagnostic; that cgroup is explicitly **not sampler-exclusive**. No per-PID PSI
is claimed. GPU identity/ownership, thirty GiB total-device use, a twenty-eight GiB
PyTorch allocator allowance, frozen protected identities, the absolute lease,
eight GiB disk reserve/output cap and parent-death cleanup remain enforced.
The memory loop normally runs each second and NVML every five seconds; external
queries can delay a poll. A query timeout is unknown telemetry, with a finite
ninety-second grace and validated owned-progress fallback.

## Validation and limitations

CPU tests exercise real upstream routed modules: exact OFF, signed/multiple raw
sliders, base storage/RNG preservation, greedy token equality, alpha-program reuse,
checkpoint cache invalidation and old snapshots, scope execution across split and
malformed markers, Unicode, HTTP streaming/disconnect/cancel, frozen provenance,
context/output bounds, RAM/OOM cases and owned-worker cleanup. The tiny synthetic
host is not evidence of 30B behavior. A trained artifact's availability is not a
guarantee of style quality, factual fidelity, broad generalization or predictable
multi-slider blending. The ordinary-LoRA approximation remains a separate export
format; it is not required to use the raw particle adapter.

## Portable launcher native smoke

On 2026-10-30, the PR checkout loaded the pinned 30B base and fourteen verified
raw particle adapters on one admitted GPU, with the existing protected service
left running. Exact OFF logits passed. A normal answer streamed before completion
and ended at EOS after eleven tokens. A second sample emitted 115 reasoning
frames before the final answer and ended at EOS after 147 tokens. Mixed per-slider
scopes recorded the expected unknown → thinking → unknown → response → unknown
transitions. The receipt contains prompts, settings, checkpoint hashes and final
answers, without distributing weights or private configuration.

The first attempt uncovered a busy-port packaging bug; its owned model was
cleaned up, and the launcher now reserves the port before loading. The regression
is tested. The corrected smoke completed in 113.274 seconds; both attempts used
192.629 aggregate guarded GPU-seconds. All three locks and the owned allocation
were released, the protected process stayed unchanged, and no host OOM occurred.
This checks native startup, streaming and routing integration on one environment;
it does not establish broad style quality, channel isolation or training stability.
The earlier README screenshot remains the explicitly labeled model-free preview.
