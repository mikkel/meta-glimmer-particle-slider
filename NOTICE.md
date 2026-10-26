# Upstream attribution

This integration uses HyperGAN/particle-sliders at commit
`1f5b380d317b73779a8aeab63b6bec0093c2c65c` under its MIT license
(copyright 2023 Rohit Gandikota, preserved in LICENSE):

- `conceptmod/textsliders/particle_bridge_gan.py`: original paired-error ParticleGAN update, critic and sampler.
- `conceptmod/textsliders/lora.py`: ordinary LoRA modules and application.
- `packages/particle-sliders-core/src/particle_sliders/formulation.py`: current formulation stamp and routed bridge.
- `packages/particle-sliders-core/src/particle_sliders/distillation.py`: unchanged ridge fitter.

ParticleGAN at `82b72661c03377f348ccb76de37cd4ca02d45a8d` supplies the current
core GAN primitives. Its source and license remain with that dependency.

The sampling and export structure follows HyperGAN/anima-particle-sliders
commit `36ad126761056fbb06d6b02574149f0634f50033`, specifically
`lumen_studio/particles.py` and `release_tools/distill.py`. Its image model
code, checkpoints, and model-specific training data are not included.

Muse Glimmer weights are a separate Meta release under Apache-2.0. No weights
or upstream trained artifacts are distributed by this repository.
