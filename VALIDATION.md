# CPU verification

Command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q --upstream=/path/to/particle-sliders
```

The implementation was tested using Python 3.12, Torch 2.13.0+cu126 executing
CPU operations, safetensors 0.8.0, pytest 9.1.1, upstream particle-sliders
`1f5b380d317b73779a8aeab63b6bec0093c2c65c`, and ParticleGAN
`82b72661c03377f348ccb76de37cd4ca02d45a8d` installed from its Git source.

Seventeen tests pass. The synthetic causal fixture exercises actual upstream
adversarial updates and the current core routed bridge. Coverage includes:

- Grouped query/KV widths, text-only projection selection, live bank routing,
  fixed-bank sampling, additive strength and exact zero bypass.
- Frozen host parameters and changes to trainable particle/branch parameters.
- Four uninterrupted updates equal two updates plus save/reload plus two
  updates, including adapter, discriminator, EMA, sampler and held-out metrics.
- A trained/reloaded teacher at zero strength equals the unattached baseline.
- Strict artifact identity checks and rollback after a corrupt teacher fails.
- Actual upstream ridge calls, retained teacher U matrices and ordinary LoRA
  A/B reload with no particle bank or router in the student artifact.
- Numerical equivalence of the original AR callback game's losses,
  normalization, gmix critic, VIC and b_cap with the current core stamp.
- The train/resume/sample/distill command flow with an explicitly synthetic
  host, fixed generation protocol and immutable checkpoint/metric outputs.
- Tokenizer mapping/list/tensor handling, prompt admission before model load,
  local model byte/RAM admission and output traversal/overwrite/reserve guards.

The CLI host loader and tokenizer service are mocked in the command-flow test;
this is not native Glimmer integration evidence. No model weights are part of
the fixture, and no model load, GPU allocation, native steering result, or
held-out distillation quality result is claimed. The pinned official model
revision is documented by Meta's post-training recipe; directory naming alone
does not verify that an independently staged snapshot contains those weights.

The pinned public tokenizer_config.json was read directly to verify the native
response parser's `start_anchor` and `to=self`/`to=user` fields (79,936 bytes,
SHA256 `781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5`).
Only metadata was fetched; parsing still requires native integration testing.
