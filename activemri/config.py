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
    visualization_dir: Optional[pathlib.Path] = None
    checkpoint: Optional[str] = None
    pretrained_recon: Optional[str] = None

    # Mask / sampler
    accelerations: List[float] = field(default_factory=lambda: [4.0])
    sampler_type: str = 'conv_2d'
    slope: float = 10.0
    sampler_chans: int = 64
    sampler_depth: int = 4
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
    freeze_recon: bool = False
    fixed_mask_checkpoints: Optional[Dict[str, str]] = None  # anatomy -> loupe_2d checkpoint whose learned mask is copied into that anatomy's slot and frozen; needs anatomy_conditioning
    sense_mask_method: Optional[str] = 'pre_network'
    with_recon: bool = False
    with_gfactor: bool = False
    gfactor_replicas: int = 16
    uncertainty_mode: str = 'none'

    # Data
    min_energy_fraction: float = 0.2

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
