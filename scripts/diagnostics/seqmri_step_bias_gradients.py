"""Which steps' seqmri_2d logit maps get gradient? One training forward and backward pass of fscr1's trained F and its
data-blind twin with zero per-step maps added (`sampler.seqmri_step_bias`), on a few FLAIR dev slices, printing the
gradient norm each step's map receives. SeqMRI's loop detaches the mask between steps, so a step's draw is credited only
through what later steps read; the twin reads no reconstruction.

    python scripts/diagnostics/seqmri_step_bias_gradients.py --runs-dir /export/scratch/bkadota/multistep_adaptivity_sync/runs
"""
import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval')]

import torch  # noqa: E402

import eval_multistep_adaptivity as ev  # noqa: E402
from activemri.training.trainer import build_model  # noqa: E402
from activemri.utils.multistep import step_thresholds  # noqa: E402


def biased_model(run, device, union_route=False):
    checkpoint = torch.load(run / 'last_model.pt', map_location='cpu', weights_only=False)
    options = copy.deepcopy(checkpoint['options'])
    options.seqmri_step_bias = True
    options.seqmri_union_route = union_route
    model = build_model(options)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    if unexpected or missing != ['sampler.mask_net.step_bias']:
        raise SystemExit(f'{run.name}: missing {missing}, unexpected {unexpected}')
    return model.to(device).train(), options


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', default='/export/scratch/bkadota/multistep_adaptivity_sync/runs')
    parser.add_argument('--data-root', default='hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}')
    parser.add_argument('--slices', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--union-route', action='store_true', help='also keep every step\'s increment in the union')
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    ev_args = argparse.Namespace(workers=0, val_half=None, data_root=[args.data_root], chunk=8, limit=None,
                                 draw_seeds=[1234], device=args.device, figures=0, gradients=True)
    cfg = ev.datasets_config(runs / 'F_4_s42_fscr1_tF', ev_args)
    cfg.datasets.train_tier = 'S'
    _, val_loader = ev.build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn
    indices = list(range(0, len(dataset), max(1, len(dataset) // args.slices)))[:args.slices]
    for name in ('F_4_s42_fscr1_tF', 'Fb_4_s42_fscr1_tF'):
        model, options = biased_model(runs / name, args.device, args.union_route)
        part = ev.labelled(collate([dataset[i] for i in indices]), options, args.device)
        thresholds = step_thresholds(1234, indices, model.num_step, model.shape, args.device)
        pred = model(part['target'], part['kspace'], anatomy_labels=part['labels'], thresholds=thresholds)
        loss, _ = model.loss(pred, {'target': part['target'], 'kspace': part['kspace']}, ev.LOSS_META, options.loss_type)
        loss.backward()
        bias_grad = model.sampler.mask_net.step_bias.grad
        per_step = bias_grad.flatten(1).norm(dim=1).tolist() if bias_grad is not None else [0.0] * model.num_step
        unet = torch.stack([p.grad.norm() for n, p in model.sampler.mask_net.named_parameters()
                            if n != 'step_bias' and p.grad is not None]).norm().item()
        print(f"{name}{' (union route)' if args.union_route else ''}: loss {loss.item():.5f}; |grad| of step 0-{model.num_step - 1} maps: "
              + ', '.join(f'{value:.3e}' for value in per_step) + f"; sampler U-Net |grad| {unet:.3e}", flush=True)


if __name__ == '__main__':
    main()
