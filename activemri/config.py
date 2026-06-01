import pathlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # Model architecture
    num_pools: int = 4
    num_step: int = 2
    drop_prob: float = 0.0
    num_chans: int = 64
    model: str = ''
    input_chans: int = 1
    output_chans: int = 2

    # Training
    batch_size: int = 4
    num_epochs: int = 50
    lr: float = 0.001
    lr_step_size: int = 40
    lr_gamma: float = 0.1
    weight_decay: float = 0.0
    report_interval: int = 100
    limit_batches: float = 1.0

    # Loss
    loss_type: str = 'l1'
    recon_weight: float = 1.0
    entropy_weight: float = 0.0
    sparsity_weight: float = 0.0
    kspace_weight: float = 0.0
    uncertainty_loss: bool = False
    uncertainty_weight: float = 0.0

    # Noise
    noise_type: str = 'none'
    noise_level: float = 5e-5

    # Device / parallelism
    device: str = 'cuda'
    data_parallel: bool = False

    # Paths
    exp_dir: pathlib.Path = pathlib.Path('checkpoints')
    visualization_dir: Optional[pathlib.Path] = None
    data_path: Optional[str] = None
    checkpoint1: Optional[str] = None
    resume: bool = False

    # Data
    dataset_name: str = ''
    sample_rate: float = 1.0
    resolution: List[int] = field(default_factory=lambda: [128, 128])
    label_range: Optional[List[int]] = None
    seed: int = 42

    # Mask / sampler
    accelerations: List[float] = field(default_factory=lambda: [4.0])
    line_constrained: bool = False
    conjugate_mask: bool = False
    preselect: bool = False
    preselect_num: int = 2
    bi_dir: bool = False
    binary_sampler: bool = False
    clamp: float = 1e10

    # Reconstruction
    unet: bool = False
    old_recon: bool = False
    detach_kspace: bool = False
    pretrained_recon: bool = False
    recon_type: str = 'unet'
    cg_iters: int = 10
    cg_lamda: float = 0.0
    sense_tikhonov_weight: float = 0.0

    # Misc
    random_rotate: bool = False
    fixed_input: bool = False
    save_model: bool = False
    test: bool = False
    test_visual_frequency: int = 1000
