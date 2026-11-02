import numpy as np
import SimpleITK as sitk

from scripts.plots.subject001_recons import SCANS, load_volume

SPACING_ZYX = (184 / 160, 1.0, 1.0)
REFERENCE = "GRAPPA"


def to_image(volume):
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(SPACING_ZYX[::-1])
    return image


def register_rigid(fixed, moving):
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsCorrelation()
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=0)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SetInitialTransform(initial, inPlace=False)
    return method.Execute(fixed, moving), method


def coregister_scans():
    volumes = {name: load_volume(mid) for name, mid in SCANS.items()}
    fixed = to_image(volumes[REFERENCE])
    aligned = {REFERENCE: volumes[REFERENCE]}
    for name, volume in volumes.items():
        if name == REFERENCE:
            continue
        moving = to_image(volume)
        transform, method = register_rigid(fixed, moving)
        euler = sitk.Euler3DTransform(transform.GetNthTransform(0)) if hasattr(transform, "GetNthTransform") else transform
        print(name, "params (rx,ry,rz rad; tx,ty,tz mm):", np.round(euler.GetParameters(), 4),
              "| final metric", round(method.GetMetricValue(), 4))
        resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)
        aligned[name] = sitk.GetArrayFromImage(resampled)
    return volumes, aligned


if __name__ == "__main__":
    volumes, aligned = coregister_scans()
    for name in SCANS:
        a, b = volumes[REFERENCE].ravel(), volumes[name].ravel()
        c, d = aligned[REFERENCE].ravel(), aligned[name].ravel()
        print(name, "corr with ref before/after:", round(np.corrcoef(a, b)[0, 1], 4), round(np.corrcoef(c, d)[0, 1], 4))
    np.savez_compressed("/tmp/subject001_aligned.npz", **{k: v.astype(np.float32) for k, v in aligned.items()})
