"""Fixed variable-density Poisson-disc undersampling mask (the compressed-sensing baseline), from SigPy."""
import numpy as np
import sigpy.mri
import torch


def poisson_mask(shape, acceleration, calibration_size, seed=0):
    """Binary `[H, W]` SigPy Poisson-disc mask in DC-centred layout, with a fully sampled calibration block.

    SigPy hits the requested acceleration to within its tolerance, so the sample count is approximate.
    """
    mask = sigpy.mri.poisson(tuple(shape), acceleration, calib=(calibration_size, calibration_size), seed=seed)
    return torch.from_numpy(np.abs(mask).astype(np.float32))
