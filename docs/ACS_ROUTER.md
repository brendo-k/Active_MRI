# Simple ACS mask router

`cnn_router_2d` uses only the central ACS to select one learned mask. It does not use anatomy labels.

Architecture: mean-centered log ACS magnitude → two 3×3 convolution/ReLU layers → global average pooling → linear layer → mask index.

The mask bank and CNN train together using the existing reconstruction loss. Training uses hard Gumbel-softmax: one mask is selected in the forward pass, with a soft gradient to the CNN. Evaluation uses deterministic argmax. The selected LOUPE probability map goes through the existing budget rescaling and Bernoulli sampling, so the *index* is deterministic at evaluation; the binary acquisition mask still uses the sampler's seeded draw.

Start with two masks and 32 CNN channels:

```bash
rtk proxy .venv/bin/python train.py --model sequential --recon varnet \
  --dataset brain_knee_balanced --acceleration 8 \
  sampler.sampler_type=cnn_router_2d sampler.bank_size=2 \
  sampler.adapt_hidden=32 sampler.anatomy_conditioning=false \
  --out-dir checkpoints/cnn_router_r8
```

The ACS size is controlled by `sampler.preselect_num` (default 10). Bank entries start as independent learnable LOUPE maps; index 0/1 has no predefined anatomical meaning. This does not automatically load existing anatomy masks.

`model.sampler.mask_net.indices(observed_kspace)` returns deterministic indices for NHWC observations. The standalone `ACSRouter` accepts normalized magnitude patches `[B, 1, C, C]` and returns integer indices `[B]`; its `logits` method exposes scores for training.

This is one decision per supplied example. It does not add volume aggregation or sequential feedback. No training run or performance claim accompanies the implementation.
