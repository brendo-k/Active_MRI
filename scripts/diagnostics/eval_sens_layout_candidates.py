"""Paired held-out evaluation of full-resolution implementation changes.

Same trained weights and acquisition for every arm. This measures inference
perturbations, not convergence after retraining with changed rounding.
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ.setdefault('OMP_NUM_THREADS', '8')
import varnet_throughput_probe as probe
from sens_layout_candidates import replace_sensitivity_norms, ChannelsLastBody
import torch
from omegaconf import OmegaConf
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model, compile_unets
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.metrics import MaskedMetricAccumulator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['baseline', 'group', 'moments', 'fused'])
    parser.add_argument('--cudnn-limit', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = args.cudnn_limit
    ckpt = torch.load(args.run/'last_model.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.load(args.run/'.hydra/config.yaml')
    cfg.training.batch_size = 16
    cfg.datasets.num_workers = 4
    _, loader = build_dataloaders(cfg)
    originals = (torch.fft.fftshift, torch.fft.ifftshift)
    probe.install_fused_shifts()
    fused = (torch.fft.fftshift, torch.fft.ifftshift)
    torch.fft.fftshift, torch.fft.ifftshift = originals
    models = {}
    for arm in args.arms:
        model = build_model(ckpt['options']).cuda().eval()
        model.load_state_dict(ckpt['model'])
        if arm in ('group', 'moments'):
            replace_sensitivity_norms(model.sensetivity_net.model, arm)
            model.sensetivity_net.model.to(memory_format=torch.channels_last)
        compile_unets(model)
        if arm in ('group', 'moments'):
            model.sensetivity_net.model = ChannelsLastBody(model.sensetivity_net.model)
        models[arm] = model
    rows = {arm: {'psnr': [], 'ssim': []} for arm in args.arms}
    anatomies, volumes = [], []
    max_diff = {arm: 0.0 for arm in args.arms}
    for index, batch in enumerate(loader):
        kspace, target = batch['kspace'].cuda(), batch['target'].cuda()
        metric_mask = metric_mask_from_real_kspace(kspace, batch['anatomy'])
        for arm, model in models.items():
            torch.fft.fftshift, torch.fft.ifftshift = fused if arm == 'fused' else originals
            output = model(target, kspace)['output']
            if arm == 'baseline':
                reference = output.clone()
            max_diff[arm] = max(max_diff[arm], float((output-reference).abs().max()))
            metric = MaskedMetricAccumulator(device='cuda')
            metric.update(output, target, metric_mask)
            rows[arm]['psnr'].extend(metric.psnr_values)
            rows[arm]['ssim'].extend(metric.ssim_values)
        anatomies.extend(batch['anatomy'])
        volumes.extend(batch['fname'])
        print(f'batch {index+1}/{len(loader)}', flush=True)
    summary = {'checkpoint': str(args.run/'last_model.pt'), 'epoch': ckpt['epoch'],
               'cudnn_benchmark_limit': args.cudnn_limit,
               'scope': 'same weights; no retraining; masked pixel PSNR/SSIM, slice averaged',
               'max_output_abs_difference': max_diff, 'anatomies': {}}
    for anatomy in sorted(set(anatomies)):
        keep = [i for i, a in enumerate(anatomies) if a == anatomy]
        summary['anatomies'][anatomy] = {'slices': len(keep), 'volumes': len({volumes[i] for i in keep}), 'arms': {}}
        for arm in args.arms:
            stats = {}
            for metric in ('psnr', 'ssim'):
                values = torch.tensor([rows[arm][metric][i] for i in keep], dtype=torch.float64)
                base = torch.tensor([rows['baseline'][metric][i] for i in keep], dtype=torch.float64)
                delta = values-base
                stats[metric] = {'mean': float(values.mean()), 'delta_mean': float(delta.mean()),
                                 'delta_min_slice': float(delta.min()), 'delta_max_slice': float(delta.max())}
            summary['anatomies'][anatomy]['arms'][arm] = stats
    args.out.write_text(json.dumps(summary, indent=2))
    args.out.with_name(args.out.stem+'_slices.json').write_text(json.dumps(
        {'anatomy': anatomies, 'volume': volumes, 'metrics': rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
