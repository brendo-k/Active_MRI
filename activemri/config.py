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
    optimizer: str = 'adam'  # adam | muon (Muon on the U-Nets' hidden convolutions, Adam on the rest)
    muon_lr: Optional[float] = None  # lr of the Muon group; None = lr (Muon updates are RMS-matched to Adam's)
    muon_momentum: float = 0.95
    grad_clip_norm: float = 1.0  # per-group global-norm clip; PromptMR+ trains with 0.01
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
    compile_unets: bool = False
    channels_last_unets: object = False  # False | True | 'sens' (the sensitivity net alone); with compile_unets: inductor runs the U-Nets channels-last (its heuristic skips them), no cuDNN layout transposes
    cudnn_benchmark: bool = True
    train_masked_metrics_every: int = 1
    train_metrics_every: int = 1  # train-batch metrics (PSNR/SSIM/NMSE, masked and per anatomy) from every Nth batch only
    val_every: int = 1  # validate every Nth epoch and at the last
    early_stop_patience: int = 0  # >0: stop after this many validations in a row without a new best val loss
    fused_adam: bool = False  # torch.optim.Adam(fused=True)
    ema_decay: float = 0.0  # >0: keep an EMA of the reconstructor and sensitivity net, saved as ema_model.pt at the end of training
    visualize_every: int = 1  # images (PNG panels, wandb images, fixed val examples) every Nth epoch and at the last

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
    selection: str = 'bernoulli'  # loupe_2d straight_through: 'bernoulli' draws, or 'dcm' = deterministic top-M of a VD prior minus the learned map (Sun et al., TCI 2024)
    dcm_vd_power: float = 4.0  # selection=dcm: exponent p of the acceptance prior ((D - r) / D)^p
    dcm_tau: float = 5.0  # selection=dcm: softmax temperature of the straight-through backward
    logit_limit: float = 6.0  # conv_2d: scores are soft-clamped to +/- this before the sigmoid
    logit_leak: float = 0.01  # conv_2d: slope of the score beyond the clamp (keeps the gradient of pinned locations alive)
    flat_mask_fraction: float = 0.0  # loupe_2d straight-through: fraction of training batches masked by a flat-probability draw at the same budget (reconstructor-only update)
    flat_mask_decay_epochs: Optional[float] = None  # flat_mask_fraction falls linearly to 0 over this many epochs (None = constant)
    adapt_grid: int = 16  # smooth_adapt_2d: side of the coarse offset grid (0 = no offset: a bounded LOUPE mask)
    adapt_hidden: int = 64  # smooth_adapt_2d / bank_adapt_2d: width of the offset predictor / gate
    bank_size: int = 8  # bank_adapt_2d: number of learned masks the calibration-block gate blends
    gate_temperature: float = 1.0  # bank_adapt_2d: gate logits are divided by this before the softmax (>1 flattens the weights)
    gate_lr: Optional[float] = None  # bank_adapt_2d: lr of the gate network (the bank's masks keep lr); None = lr for everything
    gate_logit_limit: float = 0.0  # bank_adapt_2d: >0 bounds the gate logits to +/- this (tanh), so no bank mask's weight can vanish
    init_mask_checkpoint: Optional[str] = None  # loupe_2d / smooth_adapt_2d: start every class's mask from this loupe_2d run's mask (trainable)
    loupe_vd_init_power: float = 0.0  # loupe_2d straight-through: >0 starts the mask at the budget-matched VD prior clamp(c ((D - r) / D)^p) instead of uniform noise
    loupe_same_start: bool = False  # per-anatomy loupe masks: every anatomy's mask starts from the first one's random draw
    sampler_lr: Optional[float] = None  # lr of the sampler's parameters when it differs from training.lr
    adapt_head_lr: float = 1e-4  # smooth_adapt_2d: lr of the offset predictor (the base mask keeps sampler_lr)
    sampler_chans: int = 64
    sampler_depth: int = 4
    sampler_head: str = 'none'  # conv_2d/mnet_2d: 'none', 'bias', 'mlp' or 'encoder_mlp' location-specific head on the U-Net scores
    sampler_head_mlp_lr: float = 1e-4  # sampler_head=mlp/encoder_mlp: lr of the head's MLP alone (the per-location bias keeps `lr`)
    sampler_image_input: bool = False
    caipi_keep_fraction: float = 1.0
    caipi_accel_y: int = 2
    caipi_accel_z: int = 2
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
    fixed_mask_path: Optional[str] = None  # saved [H, W] binary mask (centre included) used for every slice; the sampler is then a placeholder and only the reconstructor trains
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
    budget_over: str = 'grid'  # 'acquired': 1/R of each slice's acquired k-space; 'grid' (older runs): 1/R of the grid, spent inside the acquired region
    sampler_coil_combine: str = 'sense'  # sampler input image: 'sense' (maps' free common phase makes it phase-random) or 'rss' of the coil images
    data_blind: bool = False  # seq_adapt_2d: the twin, whose residual sees M_i, the step and the header but a fixed tensor for the data
    residual_chans: int = 16  # seq_adapt_2d: first-conv width of the residual U-Net
    residual_depth: int = 4  # seq_adapt_2d: down/up-sampling stages of the residual U-Net
    residual_limit: float = 6.0  # seq_adapt_2d: the residual is bounded to +/- this many logits (limit * tanh(. / limit))
    residual_lr: float = 1e-4  # seq_adapt_2d: lr of the residual U-Net, in its own Adam group (the static maps keep sampler_lr or lr)
    residual_checkpoint: bool = True  # seq_adapt_2d: recompute the residual U-Net in backward instead of storing every step's activations
    sampler_input: str = 'measured'  # seq_adapt_2d: 'recon' also feeds the residual a frozen reconstructor's image of y * M_i; 'full' (kscr2 clairvoyant) reads the fully sampled k-space; 'seqmri' (kscr2) also reads virtual coils of y * M_i and of the trained reconstructor's in-loop prediction
    input_recon_checkpoint: Optional[str] = None  # sampler_input='recon': the run whose reconstructor and sensitivity net, frozen, make that image
    recon_steps: List[int] = field(default_factory=lambda: [6, 9, 12, 14])  # sampler_input='seqmri': the (0-indexed) steps before which the trained reconstructor reconstructs y * M_i for the sampler
    virtual_coils: int = 4  # sampler_input='seqmri': ACS-compressed virtual coils of the measured and predicted k-space the residual reads
    seqmri_input_scale: str = 'raw'  # seqmri_2d: 'raw' real and imaginary k-space over the ACS peak (kscr3), or 'signed_log' of them (fscr1)
    data_channels: List[str] = field(default_factory=lambda: ['image', 'kspace'])  # seq_adapt_2d: the measured-data channels the residual reads; the others get the twin's fixed values
    save_epochs: List[int] = field(default_factory=list)  # also keep epoch_NNN.pt (weights only) after these 1-indexed epochs
    sense_mask_method: Optional[str] = 'pre_network'
    with_recon: bool = False
    with_gfactor: bool = False
    gfactor_replicas: int = 16
    sens_upsample_method: str = 'conv'
    sens_conv_after_upsample: bool = False
    sens_resolution: Optional[int] = None
    sens_precision: str = 'fp32'
    sens_norm: str = 'instance'  # normalisation after every sensitivity U-Net conv (activemri/reconstructors/norms.py)
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
