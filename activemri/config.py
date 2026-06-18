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

    # Device / parallelism
    device: str = 'cuda'
    data_parallel: bool = False

    # Paths
    out_dir: List[pathlib.Path] = field(default_factory=lambda: [pathlib.Path('checkpoints')])
    visualization_dir: Optional[pathlib.Path] = None
    checkpoint: Optional[str] = None

    # Mask / sampler
    accelerations: List[float] = field(default_factory=lambda: [4.0])
    line_constrained: bool = False
    preselect_num: int = 10

    # Reconstruction
    recon: ReconConfig = field(default_factory=ReconConfig)

    # Misc
    resolution: List[int] = field(default_factory=lambda: [128, 128])
    random_rotate: bool = False
    fixed_input: bool = False
    seed: int | None = None
