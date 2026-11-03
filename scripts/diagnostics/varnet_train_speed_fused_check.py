"""fused_coil_ops against the eager cascades: relative differences of the VarNet output and of every gradient (weights,
sensitivity maps, mask), in float64 and in float32 (TF32 off), next to eager-vs-eager run-to-run differences.
A 3-cascade VarNet (chans 8, pools 2) on random 32x32 data with 4 coils, one of them zero padding (reports/varnet_train_speed).

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/varnet_train_speed_fused_check.py reports/varnet_train_speed/fused_check.json
"""
import json
import sys

import torch

import _path  # noqa: F401
from activemri.reconstructors.VarNet import VarNet


def inputs(batch=2, coils=4, size=32, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    images = torch.randn(batch, coils, size, size, dtype=torch.complex128, device='cuda', generator=g)
    maps = torch.randn(batch, coils, size, size, dtype=torch.complex128, device='cuda', generator=g)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    maps[:, -1] = 0
    images[:, -1] = 0
    kspace = torch.fft.fftshift(torch.fft.fft2(images, norm='ortho'), dim=(-2, -1))
    mask = (torch.rand(batch, 1, size, size, device='cuda', generator=g, dtype=torch.float64) < 0.3).double()
    return kspace, maps, mask


def run(fused, kspace, maps, mask, dtype):
    torch.manual_seed(0)
    net = VarNet(num_cascades=3, chans=8, pools=2, fused_coil_ops=fused).cuda().to(dtype)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    maps = maps.to(complex_dtype).clone().requires_grad_(True)
    mask = mask.to(dtype).clone().requires_grad_(True)
    out = net(kspace.to(complex_dtype) * mask, maps, mask)
    out.abs().square().mean().backward()
    return {'output': out.detach(), 'weight_grads': torch.cat([p.grad.flatten() for p in net.parameters()]),
            'map_grads': maps.grad, 'mask_grads': mask.grad}


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kspace, maps, mask = inputs()
    result = {}
    for name, dtype in (('float64', torch.float64), ('float32', torch.float32)):
        eager, eager_again, fused = (run(f, kspace, maps, mask, dtype) for f in (False, False, True))
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        result[name] = {key: {'fused_vs_eager': rel(fused[key], eager[key]), 'eager_vs_eager': rel(eager_again[key], eager[key])}
                        for key in eager}
    open(sys.argv[1], 'w').write(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
