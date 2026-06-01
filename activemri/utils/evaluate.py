"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import torch
from torchmetrics import MetricCollection, MeanSquaredError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms.functional import center_crop

from activemri.utils.metrics import NMSE


def build_metrics(device="cpu") -> MetricCollection:
    return MetricCollection(
        {
            "MSE": MeanSquaredError(),
            "NMSE": NMSE(),
            "PSNR": PeakSignalNoiseRatio(data_range=1.0),
            "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0),
        }
    ).to(device)


def evaluate(args, recons_key):
    metrics = build_metrics()

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            args.predictions_path / tgt_file.name, "r"
        ) as recons:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue

            if args.acceleration and target.attrs["acceleration"] != args.acceleration:
                continue
            target_dataset = target[recons_key]
            recons_dataset = recons[recons_key]
            assert isinstance(
                target_dataset, h5py.Dataset
            ), f"{recons_key} is not a dataset in {tgt_file}"
            assert isinstance(
                recons_dataset, h5py.Dataset
            ), f"{recons_key} is not a dataset in {args.predictions_path / tgt_file.name}"
            target_data = torch.from_numpy(target_dataset[()]).float()
            recons_data = torch.from_numpy(recons_dataset[()]).float()
            target_data = center_crop(target_data, [target_data.shape[-1], target_data.shape[-1]])
            recons_data = center_crop(recons_data, [target_data.shape[-1], target_data.shape[-1]])

            data_range = target_data.max()
            metrics["PSNR"].data_range = data_range
            metrics["SSIM"].data_range = data_range

            if target_data.ndim == 3:
                target_data = target_data.unsqueeze(1)
                recons_data = recons_data.unsqueeze(1)

            metrics.update(recons_data, target_data)

    results = metrics.compute()
    return results


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    results = evaluate(args, recons_key)
    for name, value in results.items():
        print(f"{name} = {value:.4g}")
