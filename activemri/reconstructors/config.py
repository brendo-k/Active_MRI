import argparse
from dataclasses import dataclass


@dataclass
class ReconConfig:
    recon_type: str = 'unet'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--recon_type', type=str, choices=['unet', 'lls'], default='unet')

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ReconConfig':
        if args.recon_type == 'unet':
            return UNetReconConfig.from_args(args)
        elif args.recon_type == 'lls':
            return CGSenseReconConfig.from_args(args)
        else:
            raise ValueError(f"Unknown recon_type: {args.recon_type}")


@dataclass
class UNetReconConfig(ReconConfig):
    recon_type: str = 'unet'
    num_pools: int = 4
    num_chans: int = 64
    drop_prob: float = 0.0
    in_chans: int = 1
    out_chans: int = 2

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--num-pools', type=int, default=4)
        parser.add_argument('--num-chans', type=int, default=64)
        parser.add_argument('--drop-prob', type=float, default=0.0)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'UNetReconConfig':
        return cls(
            num_pools=args.num_pools,
            num_chans=args.num_chans,
            drop_prob=args.drop_prob,
        )


@dataclass
class VarNetReconConfig(ReconConfig):
    recon_type: str = 'varnet'
    num_cascades: int = 12
    chans: int = 18
    pools: int = 4
    anatomy_conditioning: bool = False
    film_conditioning: bool = False
    dc_mode: str = 'where'


@dataclass
class CGSenseReconConfig(ReconConfig):
    recon_type: str = 'cg_sense'
    cg_iters: int = 10
    cg_lamda: float = 0.0
    sense_tikhonov_weight: float = 0.0

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--cg_iters', type=int, default=10)
        parser.add_argument('--cg_lamda', type=float, default=0.0)
        parser.add_argument('--sense_tikhonov_weight', type=float, default=0.0)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'CGSenseReconConfig':
        return cls(
            cg_iters=args.cg_iters,
            cg_lamda=args.cg_lamda,
            sense_tikhonov_weight=args.sense_tikhonov_weight,
        )
