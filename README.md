# Meta Glimmer Particle Slider

Text-tower integration for [Muse Glimmer](https://dev.meta.ai/docs/muse-glimmer), using the original [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) ParticleGAN training API. This is a library and local command-line runner. It does not include model weights or a hosted inference client.

## Installation and CPU tests

For the verifier-gated **Calibration**, **Constraint Precision**, and **Independent Judgment**
candidate examples, see [strong-fit slider flows](examples/strong_fit_sliders/README.md).
They include 54 frozen synthetic cases, exact checks before typed Jev selection,
native PG history-packet preparation, and separate development/heldout comparisons.
Run their offline checks without model weights, torch, credentials or paid calls:

```bash
PYTHONPATH=examples python3 -m unittest discover -s examples/strong_fit_sliders/tests -v
PYTHONPATH=examples python3 -m strong_fit_sliders validate --slider calibration
PYTHONPATH=examples python3 -m strong_fit_sliders questions
```

These are untrained candidate workflows; native stochastic/direct-final generation,
live Jev judgments and learned-slider performance have not been qualified.

Use a dedicated environment and a CPU PyTorch installation for the fixture tests:

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
git clone https://github.com/HyperGAN/particle-sliders.git ../particle-sliders
git -C ../particle-sliders checkout 1f5b380d317b73779a8aeab63b6bec0093c2c65c
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 pytest -q --upstream=../particle-sliders
```

The loader checks the upstream commit and SHA256 of all imported algorithm files. It imports the actual upstream source; it cannot fall back to a local objective. The ParticleGAN dependency is pinned to `82b72661c03377f348ccb76de37cd4ca02d45a8d`, matching the core package at that revision. Model-specific upstream trainers with environment side effects are not imported.

## Particle training, sampling and export

Each selected language attention projection uses the same structure as Anima's particle branches:

```
base(x) + strength * U * RoutedMLP(A*x, P)
P: one shared learned 128 x 4 bank
A: input -> rank 8; U: rank 8 -> actual projection width
router width 16; branch width 48; alpha/rank = 8/8 = 1
```

`RoutedMLP` comes from the core `winning_formulation().bridge()`. It routes each input over all particles with softmax; inference uses the learned bank deterministically. It does not draw a random particle. Strength zero bypasses the entire adapter, including after trained weights are reloaded. Only `model.language_model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}` are selected. Grouped KV widths are read from the real modules. Vision and MLP projections are excluded. One concept is attached at a time; there is no multi-concept mixer.

`Trainer.step()` calls upstream `conceptmod.textsliders.particle_bridge_gan.update` directly. This is the upstream full-model prediction callback API, rather than the core feature-only `FormulationGame`, which expects rank-eight features. The adapter bridge and training settings follow the current core stamp: gmix critic (8 tokens, width 48, one layer, four heads, score bound 8), adversarial batch 8, Adam rates G=.0006/D=.0009/particles=.006, betas=(0,.999), particle VIC on 64 particles, EMA=.995. The noise schedule remains the stamp's 1,600-step schedule even when an invocation stops earlier. The original API owns D/G updates and the lazy b_cap penalty. Tests compare its losses, critic, normalization, VIC and penalty numerically with the current core stamp. Anima's model-specific generator learning-rate override and image diffusion data pipeline are not imported.

For the causal language host, fixed paired prompts share exactly the same continuation token IDs. Let `h+` be the frozen positive final hidden state and `hθ` the adapted neutral final hidden state at that history. The upstream paired-edit normalization gives `N`; the real critic input is Gaussian noise `ε`, and the fake input is `ε + N(hθ) - N(h+)`. The upstream relativistic logistic GAN losses and particle VIC remain unchanged. There is **no token KL, supervised reconstruction term, or diffusion timestep surrogate**. Only +1 is trained; negative strength is exposed mathematically but unconstrained by this training recipe.

### Three distinct files

| Artifact | Contents | Purpose |
| --- | --- | --- |
| Particle teacher, `.safetensors` | Shared bank, A/U matrices, routing networks, alpha, base identity and training provenance | Live nonlinear particle sampling; `save_adapter` / `load_adapter` |
| Training continuation, `.pt` | Adapter, discriminator, both Adam states, EMA, private row/noise/VIC sampler streams, Python/Torch RNG, update, dataset/protocol/recipe contract | Genuine continuation with `Trainer.restore`; loaded with `weights_only=True` |
| Distilled ordinary LoRA, `.safetensors` | Fitted A, unchanged U, alpha, base identity, calibration provenance | Linear `base(x) + strength * U*A*x`; no particles or router |

Portable files validate exact target dimensions, base model ID/revision, tensor names, dtype, finiteness and alpha before use. These are Glimmer integration formats, not Anima checkpoint files or a PEFT adapter directory. The ordinary LoRA is reloaded with this package's `load_adapter`; direct third-party loader compatibility is not asserted.

Distillation captures actual projection inputs and routed bottleneck targets on explicit calibration prompts, then calls the unchanged core `fit_routed_down(..., ridge_fraction=.01)`. As in [Anima's exporter](https://github.com/HyperGAN/anima-particle-sliders/blob/36ad126761056fbb06d6b02574149f0634f50033/release_tools/distill.py), it retains U and fits A with a homogeneous dual ridge solve. This is a data-dependent approximation, not an exact merge of a nonlinear teacher. Calibration defaults to at most 128 rows per projection (hard limit 512); its reported bottleneck RMS is a calibration error, not a held-out quality claim. Evaluate teacher and student on independent prompts before using the student.

## Local Glimmer commands

The optional host uses the official open-weight Transformers class and tokenizer. [Meta's model documentation](https://dev.meta.ai/docs/muse-glimmer) identifies Glimmer as a dense 30B multimodal causal transformer with Apache-2.0 weights, text/image input and text output. [The post-training recipe](https://dev.meta.ai/docs/muse-glimmer/post-training) pins `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, and documents LoRA/QLoRA support. That upstream support is distinct from qualification of this integration.

```bash
pip install -e '.[native]'
python -m glimmer_slider train --config examples/train.json
python -m glimmer_slider sample --config examples/sample.json
python -m glimmer_slider distill --config examples/distill.json
```

Edit the paths first. Snapshot loading is **local-only**, disables remote code, and requires a directory named by the pinned revision. This name is a declaration of base identity, not independent verification of the weight contents: stage and verify the official snapshot separately. It does not download weights or configure credentials. It rejects quantized, offloaded, or CUDA modules in this release. Default maximum checkpoint bytes is 2 GiB, so it intentionally refuses the full 30B checkpoint. A provisioned CPU host must explicitly set a sufficient `max_weight_bytes`; admission additionally requires three times the indexed weight bytes plus 16 GiB available host RAM. Model loading and GPU memory behavior have not been qualified here. API-only access and GGUF inference cannot supply the differentiable host this trainer requires.

The configuration declares a fixed horizon, updates for this invocation, checkpoint cadence, train/dev pairs, optional shared history token IDs, and fixed generation prompts. Prompts are checked with the tokenizer before loading model weights. Native generation uses the official chat template with `reasoning_strength="low"`, greedy decoding, a declared token budget and no constrained output vocabulary. It preserves generated token IDs, raw native text and the tokenizer's parsed response where supported. A completion reaching the token budget may still be in reasoning; it is not recorded as a successful final answer. See [Meta's prompting guide](https://dev.meta.ai/docs/muse-glimmer/prompting).

To continue training, set `resume` to a continuation checkpoint and keep the base, prepared data, evaluation protocol, seed, horizon, Torch version and upstream recipe unchanged. `updates` is additional work, bounded by the original horizon. Metrics use the same held-out states and training normalization at every checkpoint: RMS of `N(hθ) - N(h+)`, reported separately for Off, raw and EMA. This diagnostic is not the adversarial training loss. Generation evaluations likewise retain the same prompts and decoding protocol. No metric decrease is promised.

OutputStore only writes single relative filenames beneath the configured output directory, keeps an 8 GiB disk reserve, publishes files atomically and refuses overwrites. On resume an existing initial metric receipt must match exactly. Outputs, weights, environments, checkpoints and credentials are excluded from Git. There are no cleanup or resource-allocation commands.

## Verification limits

CPU tests exercise real upstream updates through a synthetic causal model with grouped KV widths, complete split-run continuation, frozen base weights, exact strength-zero behavior after training and reload, bank-dependent live sampling, teacher/student serialization, the upstream ridge fit and bounded output guards. Tokenizer return variations and native preflight are tested with explicit fixtures/mocks. No 30B model load, native generation, GPU training, quantized autograd, multimodal training or native steering improvement has been established for this code. Model/API access is never represented as successful training.

## Attribution

Code is MIT licensed; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Upstream algorithms are executed from the pinned dependency, not republished as independently originated algorithms. Model weights retain their separate Apache-2.0 license and are not included.
