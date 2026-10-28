"""Check fused permutation outputs/gradients and a full two-cascade backward."""
import argparse
import json
from pathlib import Path

import varnet_throughput_probe as probe
import torch
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.reconstructors.VarNet import VarNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    # This test deliberately covers many ranks/dim tuples/shapes in one process.
    # Keep every case compiled instead of silently falling back after eight graphs.
    torch._dynamo.config.cache_size_limit = 64
    original = (torch.fft.fftshift, torch.fft.ifftshift)
    probe.install_fused_shifts()
    fused = (torch.fft.fftshift, torch.fft.ifftshift)
    results = {'shift_checks': []}
    for shape in [(2, 4, 256, 256), (1, 3, 31, 33), (1, 2, 32, 35)]:
        for dims in [(-1, -2), None, -1]:
            for old, new in zip(original, fused):
                x = torch.randn(*shape, device='cuda', dtype=torch.complex64, requires_grad=True)
                g = torch.randn_like(x)
                a, b = old(x, dim=dims), new(x, dim=dims)
                ga = torch.autograd.grad(a, x, g)[0]
                gb = torch.autograd.grad(b, x, g)[0]
                equal = torch.equal(a, b) and torch.equal(ga, gb)
                assert equal, (shape, dims)
                results['shift_checks'].append({'shape': shape, 'dims': dims, 'bitwise_equal': equal})
    torch.fft.fftshift, torch.fft.ifftshift = original
    model = VarNet(num_cascades=2, unet_precision='bf16').cuda()
    sens = SensetivityModel_mc(2, 2, 8, precision='bf16').cuda()
    kspace = torch.randn(2, 4, 256, 256, device='cuda', dtype=torch.complex64)
    mask = (torch.rand(2, 1, 256, 256, device='cuda') < .125)
    mask[..., 123:133, 123:133] = True
    def run():
        model.zero_grad(set_to_none=True)
        sens.zero_grad(set_to_none=True)
        maps, _ = sens(kspace, mask_method='pre_network')
        output = model(kspace * mask, maps, mask)
        loss = output.abs().mean()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for m in (model, sens) for p in m.parameters() if p.grad is not None])
        return output.detach().clone(), loss.detach().clone(), grads.detach().clone()
    reference = run()
    torch.fft.fftshift, torch.fft.ifftshift = fused
    candidate = run()
    for name, a, b in zip(('output', 'loss', 'parameter_gradients'), reference, candidate):
        results[name] = {'bitwise_equal': torch.equal(a, b), 'max_abs_error': float((a-b).abs().max())}
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
