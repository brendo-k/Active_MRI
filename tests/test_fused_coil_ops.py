import pytest
import torch

from activemri.reconstructors.VarNet import VarNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled kernels are checked on the GPU")


def _inputs(batch=2, coils=4, size=32, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    images = torch.randn(batch, coils, size, size, dtype=torch.complex64, device='cuda', generator=g)
    maps = torch.randn(batch, coils, size, size, dtype=torch.complex64, device='cuda', generator=g)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    maps[:, -1] = 0
    images[:, -1] = 0
    kspace = torch.fft.fftshift(torch.fft.fft2(images, norm='ortho'), dim=(-2, -1))
    mask = (torch.rand(batch, 1, size, size, device='cuda', generator=g) < 0.3).float()
    return kspace, maps, mask


def _run(fused, kspace, maps, mask):
    torch.manual_seed(0)
    net = VarNet(num_cascades=3, chans=8, pools=2, fused_coil_ops=fused).cuda().double()
    kspace, maps = kspace.to(torch.complex128), maps.to(torch.complex128).requires_grad_(True)
    mask = mask.double().requires_grad_(True)
    out = net(kspace * mask, maps, mask)
    out.abs().square().mean().backward()
    grads = torch.cat([p.grad.flatten() for p in net.parameters()])
    return out.detach(), grads, maps.grad, mask.grad


def test_fused_coil_ops_match_eager_in_float64():
    kspace, maps, mask = _inputs()
    eager = _run(False, kspace, maps, mask)
    fused = _run(True, kspace, maps, mask)
    for name, a, b in zip(('output', 'weight grads', 'map grads', 'mask grads'), fused, eager):
        rel = ((a - b).norm() / b.norm()).item()
        assert rel < 1e-12, f'{name}: relative difference {rel:.2e}'


def test_fused_graph_is_freed_after_backward():
    kspace, maps, mask = _inputs()
    net = VarNet(num_cascades=2, chans=8, pools=2, fused_coil_ops=True).cuda().double()
    kspace, maps = kspace.to(torch.complex128), maps.to(torch.complex128).requires_grad_(True)
    mask = mask.double().requires_grad_(True)

    def step():
        net.zero_grad()
        maps.grad = mask.grad = None
        net(kspace * mask, maps, mask).abs().square().mean().backward()
        return torch.cuda.memory_allocated()

    step()
    allocated = [step() for _ in range(5)]
    assert allocated[-1] == allocated[0], f'allocated memory grew per step: {allocated}'
