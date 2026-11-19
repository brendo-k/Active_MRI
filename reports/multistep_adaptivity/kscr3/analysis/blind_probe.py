"""Does kscr3's SeqMRI arm (F) read its data channels at all? Its expected density with the real input against the same
with the four data channels (predicted and measured virtual-coil k-space, real and imaginary) set to 0, on the slide's
three slices, plus how large those inputs are outside the pre-selected centre.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

import rotate_probe as rp  # noqa: E402
from activemri.training.builders import build_dataloaders  # noqa: E402

DATA_CHANNELS = 4


@torch.no_grad()
def main():
    reference = np.load(rp.SCORES / 'T_1_s42_kscr2.npz', allow_pickle=True)
    cfg = OmegaConf.load(rp.DATA / 'multistep_kscr3/F_4_s42_kscr3/.hydra/config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn
    model, options, epoch = rp.load_run_copy(rp.DATA / 'multistep_kscr3/F_4_s42_kscr3')
    sampler = model.sampler.mask_net
    real_inputs = sampler.inputs
    seen = {}

    def recording_inputs(old_mask, context):
        inputs = real_inputs(old_mask, context)
        seen.setdefault('first', (inputs.detach(), old_mask.detach()))
        return inputs

    def blind_inputs(old_mask, context):
        inputs = real_inputs(old_mask, context)
        return torch.cat([torch.zeros_like(inputs[:, :DATA_CHANNELS]), inputs[:, DATA_CHANNELS:]], dim=1)

    print(f'F_4_s42_kscr3 epoch {epoch + 1}')
    for label, row in rp.ROWS.items():
        index = int(reference['dataset_index'][row])
        part = rp.ev.labelled(collate([dataset[index]]), options, rp.DEVICE)
        centre, valid = model._initial_masks(part['kspace'])
        region = (valid[0, 0] > 0).cpu().numpy() & (centre[0, 0] == 0).cpu().numpy()
        seen.clear()
        sampler.inputs = recording_inputs
        real = rp.expected_density(model, part['kspace'], part['labels'], index)
        sampler.inputs = blind_inputs
        blind = rp.expected_density(model, part['kspace'], part['labels'], index)
        sampler.inputs = real_inputs
        a, b = (uniform_filter(x, size=rp.SMOOTH) for x in (real, blind))
        inputs, old_mask = seen['first']
        outside = (old_mask[0, 0] == 0)
        magnitudes = inputs[0, :DATA_CHANNELS].abs()
        predicted = magnitudes[:2].amax(dim=0)[outside]
        print(f'{label:22s} density with vs without data: corr {np.corrcoef(a[region], b[region])[0, 1]:+.4f}, '
              f'mean |difference| {np.abs(real - blind)[region].mean():.5f}; step-0 predicted k-space outside the '
              f'centre, as a fraction of the ACS peak: median {predicted.median():.1e}, 99th percentile '
              f'{predicted.quantile(0.99):.1e}; the mask channel is 0 or 1', flush=True)


if __name__ == '__main__':
    main()
