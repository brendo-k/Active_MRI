# Throughput switches are exact (2026-11-02 22:50)

`equivalence.py` trains 2 short epochs (2 cascades, batch 8, limit_batches 0.1, compile off, cuDNN autotuner off,
deterministic algorithms) with the default trainer and with `training.train_metrics_every=10 training.visualize_every=5`,
on GPU 1 next to another job.

- Both runs end with bitwise-identical parameters (74 tensors, max |diff| 0.0).
- The CPU and CUDA RNG states are identical.
- Epoch losses are identical: 0.13069867342710495, 0.11259751932488547.
- A third run (eqA, defaults) ran out of GPU memory next to the other jobs and is not part of the check.
