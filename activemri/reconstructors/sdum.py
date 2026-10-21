"""Trainable SDUM (NV-Raw2Insights-MRI) reconstructor with `VarNet`'s call signature, fine-tuned under a learned mask.

Its universal conditioning is the (mask type, acceleration, acquisition type) class embedding added inside every
transformer block. With `anatomy_conditioning` each sample's anatomy picks the acquisition type; without it every
sample gets `acq_type`.
"""
from typing import Sequence

import torch

from activemri.reconstructors.raw2insights import Raw2InsightsRecon


class SDUMRecon(Raw2InsightsRecon):
    """
    Args:
        class_acq_types: Acquisition type per anatomy class, in the order of the anatomy labels the trainer passes.
        output_scale: Constant factor on the image. The released weights return fastMRI images about 6x too large (a
            fixed ratio of 0.158-0.170 between slices, anatomies and masks, measured zero-shot), so 0.165 puts the
            fine-tuning start at the right amplitude.
        anatomy_conditioning: Route each sample to `class_acq_types[label]`; needs `anatomy_labels` in forward.
        Other arguments as :class:`Raw2InsightsRecon`; the weights are trainable.
    """

    def __init__(self, repo_path: str, class_acq_types: Sequence[str], anatomy_conditioning: bool = False,
                 output_scale: float = 0.165, **kwargs):
        super().__init__(repo_path, trainable=True, **kwargs)
        self.class_acq_types = list(class_acq_types)
        self.anatomy_conditioning = anatomy_conditioning
        self.output_scale = output_scale

    def forward(self, masked_kspace, sense_maps, mask, anatomy_labels=None, **_unused):
        return self.output_scale * self._conditioned_reconstruction(masked_kspace, mask, anatomy_labels)

    def _conditioned_reconstruction(self, masked_kspace, mask, anatomy_labels):
        if not self.anatomy_conditioning:
            return self._reconstruct(masked_kspace, mask, self.acq_type)
        if anatomy_labels is None:
            raise ValueError("an anatomy-conditioned SDUM needs anatomy_labels")
        output = torch.zeros_like(masked_kspace[:, :1])
        for label in anatomy_labels.unique().tolist():
            selected = anatomy_labels == label
            output[selected] = self._reconstruct(masked_kspace[selected], mask[selected], self.class_acq_types[label])
        return output
