import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from activemri.reconstructors.config import ReconConfig


@dataclass
class TrainConfig:
    # Model architecture
    model: str = ''
    num_step: int = 2

    # Training
    batch_size: int = 4
    num_epochs: int = 50
    lr: float = 0.001
    lr_step_size: int = 40
    lr_gamma: float = 0.1
    lr_scheduler: str = 'step'  # step | constant | cosine | warmup_cosine
    lr_warmup_epochs: int = 2  # warmup_cosine only: linear ramp from 10% of lr
    weight_decay: float = 0.0
    limit_batches: float = 1.0

    # Loss
    loss_type: str = 'l1'
    uncertainty_loss: bool = False
    uncertainty_weight: float = 0.0
    anatomy_weight: float = 0.0
    anatomy_classes: List[str] = field(default_factory=list)
    condition_classes: List[str] = field(default_factory=list)  # finer than anatomy, e.g. knee-PD / knee-PDFS
    train_anatomies: List[str] = field(default_factory=list)  # subset of anatomy_classes to train on; empty = all
    anatomy_conditioning: bool = False

    # Device / parallelism
    device: str = 'cuda'
    data_parallel: bool = False

    # Paths
    out_dir: List[pathlib.Path] = field(default_factory=lambda: [pathlib.Path('checkpoints')])
    run_name: Optional[str] = None  # wandb run name; None = built from the settings that differ from the baseline (activemri/training/run_naming.py)
    visualization_dir: Optional[pathlib.Path] = None
    checkpoint: Optional[str] = None
    pretrained_recon: Optional[str] = None

    # Mask / sampler
    accelerations: List[float] = field(default_factory=lambda: [4.0])
    sampler_type: str = 'conv_2d'
    slope: float = 10.0
    mask_grad_floor: float = 0.0  # straight_through only: floor on the mask sigmoid's derivative (LOUPE and conv samplers), as a fraction of its peak
    ste_grad_floor: float = 0.0  # straight_through only: floor on the STE surrogate's derivative, as a fraction of its peak
    mask_draws: int = 1  # straight_through only: K stratified (rotated-threshold) mask draws per slice, their losses averaged; the batch is tiled K times, so memory scales by K
    logit_limit: float = 6.0  # conv_2d: scores are soft-clamped to +/- this before the sigmoid
    logit_leak: float = 0.01  # conv_2d: slope of the score beyond the clamp (keeps the gradient of pinned locations alive)
    adapt_grid: int = 16  # smooth_adapt_2d: side of the coarse offset grid (0 = no offset: a bounded LOUPE mask)
    adapt_hidden: int = 64  # smooth_adapt_2d: width of the offset predictor
    init_mask_checkpoint: Optional[str] = None  # loupe_2d / smooth_adapt_2d: start every class's mask from this loupe_2d run's mask (trainable)
    sampler_lr: Optional[float] = None  # lr of the sampler's parameters when it differs from training.lr
    adapt_head_lr: float = 1e-4  # smooth_adapt_2d: lr of the offset predictor (the base mask keeps sampler_lr)
    sampler_chans: int = 64
    sampler_depth: int = 4
    sampler_head: str = 'none'  # conv_2d/mnet_2d: 'none', 'bias' or 'mlp' location-specific head on the U-Net scores
    sampler_head_mlp_lr: float = 1e-4  # sampler_head=mlp: lr of the head's MLP alone (the per-location bias keeps `lr`)
    sampler_image_input: bool = False
    caipi_keep_fraction: float = 1.0
    preselect_num: int = 10
    mask_train_mode: str = 'straight_through'  # or 'reinforce'/'random' (loupe_2d/loupe_1d only)
    reinforce_weight: float = 1.0
    reinforce_entropy_weight: float = 0.0
    reinforce_normalize_advantages: bool = False
    reinforce_group_size: int = 0  # mask draws per slice; 0 = the whole batch is one slice
    reinforce_vd_power: float = 0.0  # >0: start the REINFORCE policy from a radial variable-density prior, log p = -power*log(r+0.05)
    reinforce_eval: str = 'sample'  # reinforce only: val mask is a seeded draw from the policy ('sample') or its top-k logits ('topk')
    reinforce_lr: Optional[float] = None  # policy (mask logits) lr; None = same as `lr`
    autosamp_init: str = 'uniform'  # autosamp_2d only: 'uniform' or 'normal' (variable density) starting positions
    autosamp_lr: float = 0.01  # autosamp_2d only: lr of the sample positions, in grid cells per step (AutoSamp's lr_enc)
    autosamp_kernel_sigma_start: float = 0.0  # autosamp_2d only: >0 takes the positions' gradient from a Gaussian surrogate of this width (cells); 0 = bilinear surrogate
    autosamp_kernel_sigma_end: float = 1.0  # autosamp_2d only: width the surrogate anneals to (geometrically); below ~1 the boundary trap returns
    autosamp_anneal_fraction: float = 0.8  # autosamp_2d only: fraction of training over which the width anneals; after it the positions freeze
    ppo_policy: Optional[str] = None  # PPOTrainer checkpoint whose frozen deterministic policy picks every mask; the sampler is then a placeholder
    mnet_accept: str = 'q_check'  # mnet_2d only: 'q_check' trains the sampler on a refined mask only if it beats the mnet mask (and random); 'always' trains on every one
    mnet_compare_random: bool = True  # mnet_2d, q_check: the refined mask must also beat a random mask scored by the frozen warm-start reconstructor
    mnet_rollback: bool = True  # mnet_2d, q_check: a rejected batch also reverts the reconstructor update made while refining the mask
    mnet_degeneracy_fallback: bool = True  # mnet_2d: refine from random masks when over half the batch's mnet masks are identical
    mnet_label_binarize: str = 'threshold'  # mnet_2d: mask-backward feeds the reconstructor 'threshold' (label probability > 0.5, as the paper) or the budgeted 'topk'
    mnet_random_retry: bool = False  # mnet_2d: if the refined mask does not beat the mnet mask, roll back and refine again from a random mask (the official code's remedy)
    mnet_label_steps: int = 20  # mnet_2d: mask-backward iterations per batch (T)
    mnet_sampler_steps: int = 40  # mnet_2d: sampler updates on the accepted labels per batch (c)
    mnet_sparsity_weight: float = 2e-5  # mnet_2d: weight of the L1 norm of the label mask in mask-backward (alpha)
    mnet_consistency_weight: float = 0.05  # mnet_2d: weight of the mean BCE tying the label mask to the sampler's output in mask-backward (lambda)
    mnet_init_margin: float = 0.1  # mnet_2d: label logits start at +margin on the mnet mask's locations, -margin elsewhere
    mnet_label_lr: Optional[float] = 0.02  # mnet_2d: lr of the label logits in mask-backward; None = same as `lr`
    freeze_recon: bool = False
    fixed_mask_checkpoints: Optional[Dict[str, str]] = None  # anatomy -> loupe_2d checkpoint whose learned mask is copied into that anatomy's slot and frozen; needs anatomy_conditioning
    sense_mask_method: Optional[str] = 'pre_network'
    with_recon: bool = False
    with_gfactor: bool = False
    gfactor_replicas: int = 16
    uncertainty_mode: str = 'none'

    # Data
    min_energy_fraction: float = 0.2
    pseudo_target: bool = False  # with ssdu: the target is the frozen reconstructor's image from all acquired data, and the loss is on images (stage 2)
    ssdu: bool = False  # k-space is already scanner-undersampled and self-supervised: the loss is on held-out acquired points, the sampler is confined to acquired points

    # Reconstruction
    recon: ReconConfig = field(default_factory=ReconConfig)

    # Misc
    resolution: List[int] = field(default_factory=lambda: [256, 256])
    random_rotate: bool = False
    phase_scramble_std: float = 0.0  # >0: multiply every coil image by a smooth random phase of this std (radians) before the model
    phase_scramble_band: List[float] = field(default_factory=lambda: [6.0, 16.0])  # cycles/FOV
    fixed_input: bool = False
    seed: int | None = None

    @property
    def label_classes(self) -> List[str]:
        """Class names the conditioning labels index into."""
        return self.condition_classes or self.anatomy_classes

    @property
    def label_key(self) -> str:
        """Batch field holding the name of each sample's conditioning class."""
        return 'condition' if self.condition_classes else 'anatomy'
