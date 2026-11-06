"""fused_coil_ops against the eager cascades at the real size: 12 cascades, 18 channels, LayerNorm, 20 coils, 320x320, in
float64. Rounding is then about 1e-16, so the relative differences of the output and of the weight, map and mask gradients
measure the arithmetic, whatever the kinks of the LeakyReLUs do to float32 or bf16 runs. Needs a GPU.

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/muon_layer_speed_fused_check.py <out.json>
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from activemri.reconstructors.VarNet import VarNet
from activemri.reconstructors.norms import replace_instance_norms

BATCH, COILS, SIZE = 2, 20, 320


def inputs():
    generator = torch.Generator(device='cuda').manual_seed(0)
    draw = lambda: torch.randn(BATCH, COILS, SIZE, SIZE, dtype=torch.complex128, device='cuda', generator=generator)
    images, maps = draw(), draw()
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    maps[:, -5:] = 0
    images[:, -5:] = 0
    kspace = torch.fft.fftshift(torch.fft.fft2(images, norm='ortho'), dim=(-2, -1))
    mask = (torch.rand(BATCH, 1, SIZE, SIZE, device='cuda', generator=generator, dtype=torch.float64) < 0.125).double()
    return kspace, maps, mask


def run(fused, kspace, maps, mask):
    torch.manual_seed(0)
    net = replace_instance_norms(VarNet(num_cascades=12, chans=18, pools=4, fused_coil_ops=fused), 'layer').cuda().double()
    maps, mask = maps.clone().requires_grad_(True), mask.clone().requires_grad_(True)
    output = net(kspace * mask, maps, mask)
    output.abs().mean().backward()
    return {'output': output.detach(), 'weight_grads': torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]),
            'map_grads': maps.grad, 'mask_grads': mask.grad}


kspace, maps, mask = inputs()
eager, fused = run(False, kspace, maps, mask), run(True, kspace, maps, mask)
result = {key: ((fused[key] - eager[key]).norm() / eager[key].norm()).item() for key in eager}
Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
