import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

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
    weight_decay: float = 0.0
    limit_batches: float = 1.0

    # Loss
    loss_type: str = 'l1'
    uncertainty_loss: bool = False
    uncertainty_weight: float = 0.0
    anatomy_weight: float = 0.0
    anatomy_classes: List[str] = field(default_factory=list)
    anatomy_conditioning: bool = False

    # Device / parallelism
    device: str = 'cuda'
    data_parallel: bool = False

    # Paths
    out_dir: List[pathlib.Path] = field(default_factory=lambda: [pathlib.Path('checkpoints')])
    visualization_dir: Optional[pathlib.Path] = None
    checkpoint: Optional[str] = None

    # Mask / sampler
    accelerations: List[float] = field(default_factory=lambda: [4.0])
    sampler_type: str = 'conv_2d'
    slope: float = 10.0
    preselect_num: int = 10
    mask_train_mode: str = 'straight_through'  # or 'reinforce' (loupe_2d/loupe_1d only)
    reinforce_weight: float = 1.0
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
    fixed_input: bool = False
    seed: int | None = None
